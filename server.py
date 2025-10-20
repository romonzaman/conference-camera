#!/usr/bin/env python3
"""
WebRTC Conference Server with Active Speaker Detection
Handles multiple client connections and switches video to active speaker
"""

import asyncio
import json
import logging
import time
import base64
from typing import Dict, Set, Optional
import numpy as np
import cv2
from aiohttp import web, ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
import av

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state management
pcs: Set[RTCPeerConnection] = set()
video_tracks: Dict[str, VideoStreamTrack] = {}
audio_levels: Dict[str, Dict] = {}
active_speaker_id: Optional[str] = None
server_video_track = None
client_info: Dict[str, Dict] = {}  # Store client name and channel info
viewer_websockets: Set[web.WebSocketResponse] = set()  # WebSocket connections for viewers

# Speaker detection state
last_speaker_change_time = 0
speaker_stability_delay = 2.0  # Wait 2 seconds before switching speakers

# Configuration
AUDIO_THRESHOLD = 500
TIME_WINDOW = 1.0  # seconds
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# STUN server configuration
RTC_CONFIGURATION = RTCConfiguration(
    iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
)


class VideoTransformTrack(VideoStreamTrack):
    """
    Video track that switches between different client video streams
    based on active speaker detection
    """
    
    def __init__(self):
        super().__init__()
        self.current_track = None
        self.blank_frame = None
        self._create_blank_frame()
    
    def _create_blank_frame(self):
        """Create a blank YUV420P frame"""
        # Create a proper video frame using OpenCV
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        frame.fill(128)  # Gray background
        
        # Add some text to indicate no active speaker
        cv2.putText(frame, "No Active Speaker", (50, VIDEO_HEIGHT//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to proper video frame format
        try:
            self.blank_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        except Exception as e:
            logger.error(f"Error creating blank frame: {e}")
            # Fallback: create a simple frame
            self.blank_frame = None
    
    def set_active_speaker(self, speaker_id: str):
        """Switch to active speaker's video track"""
        logger.info(f"Setting active speaker: {speaker_id}")
        logger.info(f"Available video tracks: {list(video_tracks.keys())}")
        
        if speaker_id and speaker_id in video_tracks:
            self.current_track = video_tracks[speaker_id]
            logger.info(f"Switched to speaker: {speaker_id}")
        else:
            self.current_track = None
            if speaker_id:
                logger.info(f"Speaker {speaker_id} not found in video tracks")
            else:
                logger.info("No active speaker, showing blank frame")
    
    async def recv(self):
        """Receive next video frame"""
        if self.current_track:
            try:
                frame = await self.current_track.recv()
                logger.debug(f"Received frame from active speaker")
                
                # Broadcast frame to WebSocket viewers
                await self.broadcast_frame_to_viewers(frame)
                
                return frame
            except Exception as e:
                logger.error(f"Error receiving frame from active speaker: {e}")
                self.current_track = None
        
        # If no active speaker, try to get any available video track
        if not self.current_track and video_tracks:
            # Get the first available video track
            first_client = list(video_tracks.keys())[0]
            self.current_track = video_tracks[first_client]
            logger.info(f"Using first available video track from {first_client}")
            try:
                frame = await self.current_track.recv()
                logger.debug(f"Received frame from fallback track")
                
                # Broadcast frame to WebSocket viewers
                await self.broadcast_frame_to_viewers(frame)
                
                return frame
            except Exception as e:
                logger.error(f"Error receiving frame from fallback track: {e}")
                self.current_track = None
        
        # Return blank frame if no video available
        logger.debug("No video available, returning blank frame")
        if self.blank_frame:
            return self.blank_frame
        else:
            # Create a simple fallback frame
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
            frame.fill(128)  # Gray background
            try:
                return av.VideoFrame.from_ndarray(frame, format="bgr24")
            except Exception as e:
                logger.error(f"Error creating fallback frame: {e}")
                # Return a basic frame
                return av.VideoFrame.from_ndarray(frame, format="rgb24")
    
    async def broadcast_frame_to_viewers(self, frame):
        """Broadcast video frame to WebSocket viewers"""
        if not viewer_websockets:
            return
        
        try:
            # Convert frame to JPEG
            frame_array = frame.to_ndarray(format="bgr24")
            _, buffer = cv2.imencode('.jpg', frame_array, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Send to all connected viewers
            message = json.dumps({
                'type': 'video_frame',
                'data': frame_data,
                'timestamp': time.time()
            })
            
            # Send to all viewers concurrently
            disconnected_viewers = set()
            for ws in viewer_websockets:
                try:
                    await ws.send_str(message)
                except Exception as e:
                    logger.error(f"Error sending to viewer: {e}")
                    disconnected_viewers.add(ws)
            
            # Remove disconnected viewers
            for ws in disconnected_viewers:
                viewer_websockets.discard(ws)
                
        except Exception as e:
            logger.error(f"Error broadcasting frame: {e}")




def force_codec(pc, sender, forced_codec):
    """Force specific codec for better compatibility"""
    kind = forced_codec.split('/')[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences([codec for codec in codecs if codec.mimeType == forced_codec])


async def cleanup_client(client_id: str):
    """Clean up resources for disconnected client"""
    global active_speaker_id
    
    try:
        # Remove from video tracks
        if client_id in video_tracks:
            del video_tracks[client_id]
        
        # Remove from audio levels
        if client_id in audio_levels:
            del audio_levels[client_id]
        
        # Remove from client info
        if client_id in client_info:
            client_name = client_info[client_id]['name']
            del client_info[client_id]
            logger.info(f"Client {client_name} ({client_id}) disconnected")
        
        # Update active speaker if this was the active speaker
        if active_speaker_id == client_id:
            active_speaker_id = None
            logger.info(f"Active speaker {client_id} disconnected")
        
        logger.info(f"Cleaned up client: {client_id}")
    except Exception as e:
        logger.error(f"Error cleaning up client {client_id}: {e}")


async def monitor_audio_track(track, client_id):
    """Monitor audio track and calculate levels"""
    try:
        while True:
            try:
                frame = await track.recv()
                
                # Convert audio frame to numpy array
                audio_data = frame.to_ndarray()
                
                # Calculate RMS (Root Mean Square) for audio level
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)  # Convert to mono
                
                rms = np.sqrt(np.mean(audio_data**2))
                audio_level = int(rms * 1000)  # Scale for better visibility
                
                # Update audio levels with timestamp
                current_time = time.time()
                audio_levels[client_id] = {
                    'level': audio_level,
                    'timestamp': current_time
                }
                
                # Check for mute (very low audio level)
                if audio_level < 10:  # Very low threshold for mute detection
                    logger.debug(f"Client {client_id} appears muted (level: {audio_level})")
                
                logger.debug(f"Audio level for {client_id}: {audio_level}")
                
            except Exception as e:
                logger.error(f"Error processing audio frame for {client_id}: {e}")
                break
                
    except Exception as e:
        logger.error(f"Audio monitoring ended for {client_id}: {e}")


def is_client_muted(client_id):
    """Check if a client is muted (very low audio level)"""
    if client_id not in audio_levels:
        return True
    
    current_time = time.time()
    audio_data = audio_levels[client_id]
    
    # Check if audio is recent and above mute threshold
    if current_time - audio_data['timestamp'] > TIME_WINDOW:
        return True
    
    return audio_data['level'] < 10  # Very low threshold for mute


async def detect_active_speaker():
    """Detect active speaker based on audio levels with stability logic"""
    global active_speaker_id, last_speaker_change_time
    
    current_time = time.time()
    valid_speakers = {}
    
    # Find speakers with recent audio (within TIME_WINDOW)
    for client_id, audio_data in audio_levels.items():
        if current_time - audio_data['timestamp'] <= TIME_WINDOW:
            valid_speakers[client_id] = audio_data['level']
    
    logger.debug(f"Audio levels: {audio_levels}")
    logger.debug(f"Valid speakers: {valid_speakers}")
    
    # Find speaker with highest audio level above threshold
    max_level = 0
    new_active_speaker = None
    
    for client_id, level in valid_speakers.items():
        logger.debug(f"Client {client_id}: level={level}, threshold={AUDIO_THRESHOLD}")
        if level > AUDIO_THRESHOLD and level > max_level:
            max_level = level
            new_active_speaker = client_id
    
    # Stability logic: only switch if there's a clear new speaker and enough time has passed
    if new_active_speaker != active_speaker_id:
        time_since_last_change = current_time - last_speaker_change_time
        
        # If we have a current speaker and they're still talking, don't switch unless:
        # 1. Current speaker is muted or silent
        # 2. New speaker is significantly louder
        # 3. Enough time has passed since last change
        
        should_switch = False
        
        if active_speaker_id is None:
            # No current speaker, switch to any speaker above threshold
            should_switch = True
        elif is_client_muted(active_speaker_id) or (active_speaker_id in valid_speakers and valid_speakers[active_speaker_id] < 50):
            # Current speaker is muted or silent, switch to new speaker
            should_switch = True
        elif new_active_speaker and max_level > valid_speakers.get(active_speaker_id, 0) * 1.5:
            # New speaker is significantly louder (50% more), switch after delay
            should_switch = time_since_last_change > speaker_stability_delay
        elif time_since_last_change > speaker_stability_delay * 2:
            # Long delay, allow switching to any new speaker
            should_switch = True
        
        if should_switch:
            old_speaker = active_speaker_id
            active_speaker_id = new_active_speaker
            last_speaker_change_time = current_time
            
            if active_speaker_id:
                logger.info(f"Active speaker changed to: {active_speaker_id} (level: {max_level})")
            else:
                logger.info("No active speaker - all speakers silent")
        else:
            logger.debug(f"Speaker switch delayed - current: {active_speaker_id}, new: {new_active_speaker}, time: {time_since_last_change:.1f}s")


async def offer_handler(request):
    """Handle WebRTC offer from client"""
    try:
        data = await request.json()
        offer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
        client_id = data.get('clientId', f'client_{int(time.time())}')
        user_name = data.get('userName', 'Unknown')
        channel = data.get('channel', 0)
        
        # Store client information
        client_info[client_id] = {
            'name': user_name,
            'channel': channel,
            'connected_at': time.time()
        }
        
        logger.info(f"Received offer from client: {client_id} (Name: {user_name}, Channel: {channel})")
        
        # Create peer connection
        pc = RTCPeerConnection(RTC_CONFIGURATION)
        pcs.add(pc)
        
        # Create or get global video transform track
        global server_video_track
        if server_video_track is None:
            server_video_track = VideoTransformTrack()
        
        # Handle connection state changes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state for {client_id}: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                try:
                    await cleanup_client(client_id)
                    pcs.discard(pc)
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
        
        # Handle incoming tracks
        @pc.on("track")
        def on_track(track):
            logger.info(f"Track received from {client_id}: {track.kind}")
            logger.info(f"Track readyState: {track.readyState}")
            
            if track.kind == "video":
                # Store video track
                video_tracks[client_id] = track
                logger.info(f"Video track stored for {client_id}. Total video tracks: {len(video_tracks)}")
                
                # If this is the first video track, set it as active speaker for testing
                if len(video_tracks) == 1:
                    global active_speaker_id
                    active_speaker_id = client_id
                    logger.info(f"Set first video track as active speaker: {client_id}")
                    # Also update the video transform track immediately
                    if server_video_track:
                        server_video_track.set_active_speaker(client_id)
                
            elif track.kind == "audio":
                logger.info(f"Audio track received from {client_id} - starting monitoring")
                
                # Connect the incoming audio track to our monitoring track
                @track.on("ended")
                def on_ended():
                    logger.info(f"Audio track ended for {client_id}")
                
                # Start monitoring the audio track
                asyncio.create_task(monitor_audio_track(track, client_id))
                logger.info(f"Audio monitoring started for {client_id}")
        
        # Add video transform track to peer connection
        pc.addTrack(server_video_track)
        
        # Handle the offer
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Start active speaker detection loop
        asyncio.create_task(active_speaker_detection_loop(server_video_track))
        
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            })
        )
        
    except Exception as e:
        logger.error(f"Error handling offer: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return web.Response(status=500, text="Internal Server Error")


async def active_speaker_detection_loop(video_transform):
    """Background task for active speaker detection"""
    while True:
        try:
            await detect_active_speaker()
            video_transform.set_active_speaker(active_speaker_id)
            await asyncio.sleep(0.1)  # Check every 100ms
        except Exception as e:
            logger.error(f"Error in speaker detection loop: {e}")
            await asyncio.sleep(1)


async def status_handler(request):
    """Return current system status"""
    try:
        current_time = time.time()
        
        # Clean up old audio levels
        for client_id in list(audio_levels.keys()):
            if current_time - audio_levels[client_id]['timestamp'] > TIME_WINDOW:
                del audio_levels[client_id]
        
        # Prepare status data
        status_data = {
            "connected_clients": len(pcs),
            "video_tracks": len(video_tracks),
            "active_speaker": active_speaker_id,
            "audio_levels": {
                client_id: data['level'] 
                for client_id, data in audio_levels.items()
                if current_time - data['timestamp'] <= TIME_WINDOW
            },
            "audio_levels_raw": audio_levels,  # Include raw data for debugging
            "client_info": client_info,  # Include client information
            "timestamp": current_time
        }
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(status_data, indent=2)
        )
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return web.Response(status=500, text="Internal Server Error")


async def debug_handler(request):
    """Debug endpoint to check audio levels"""
    try:
        current_time = time.time()
        debug_data = {
            "audio_levels": audio_levels,
            "video_tracks": list(video_tracks.keys()),
            "active_speaker": active_speaker_id,
            "connected_pcs": len(pcs),
            "current_time": current_time,
            "threshold": AUDIO_THRESHOLD,
            "time_window": TIME_WINDOW,
            "client_info": client_info,
            "server_video_track_exists": server_video_track is not None
        }
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(debug_data, indent=2)
        )
        
    except Exception as e:
        logger.error(f"Error in debug handler: {e}")
        return web.Response(status=500, text="Internal Server Error")


async def viewer_handler(request):
    """Serve the server video viewer page"""
    try:
        with open('static/viewer.html', 'r') as f:
            content = f.read()
        return web.Response(content_type="text/html", text=content)
    except FileNotFoundError:
        return web.Response(
            status=404, 
            text="<h1>404 - File not found</h1><p>Please ensure static/viewer.html exists</p>"
        )


async def video_websocket_handler(request):
    """WebSocket handler for video streaming to viewers"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Add to viewer websockets
    viewer_websockets.add(ws)
    logger.info(f"Viewer connected. Total viewers: {len(viewer_websockets)}")
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get('type') == 'ping':
                    await ws.send_str(json.dumps({'type': 'pong'}))
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Remove from viewer websockets
        viewer_websockets.discard(ws)
        logger.info(f"Viewer disconnected. Total viewers: {len(viewer_websockets)}")
    
    return ws


async def video_feed_handler(request):
    """Serve the server's video feed as a simple stream"""
    try:
        # For now, return a simple message indicating the video feed
        # In a real implementation, this would stream the video
        return web.Response(
            content_type="text/html",
            text="""
            <html>
            <head><title>Server Video Feed</title></head>
            <body>
                <h1>Server Video Feed</h1>
                <p>This would show the active speaker's video.</p>
                <p>Active Speaker: {active_speaker_id}</p>
                <p>Video Tracks: {video_tracks_count}</p>
                <p><a href="/viewer">Back to Viewer</a></p>
            </body>
            </html>
            """.format(
                active_speaker_id=active_speaker_id or "None",
                video_tracks_count=len(video_tracks)
            )
        )
    except Exception as e:
        logger.error(f"Error in video feed handler: {e}")
        return web.Response(status=500, text="Internal Server Error")


async def friends_handler(request):
    """Serve the friends list page"""
    try:
        with open('static/friends.html', 'r') as f:
            content = f.read()
        return web.Response(content_type="text/html", text=content)
    except FileNotFoundError:
        return web.Response(
            status=404, 
            text="<h1>404 - File not found</h1><p>Please ensure static/friends.html exists</p>"
        )


async def index_handler(request):
    """Serve the main HTML page"""
    try:
        with open('static/index.html', 'r') as f:
            content = f.read()
        return web.Response(content_type="text/html", text=content)
    except FileNotFoundError:
        return web.Response(
            status=404, 
            text="<h1>404 - File not found</h1><p>Please ensure static/index.html exists</p>"
        )


async def on_shutdown(app):
    """Cleanup on server shutdown"""
    logger.info("Shutting down server...")
    
    # Close all peer connections
    for pc in pcs:
        await pc.close()
    
    # Clear global state
    pcs.clear()
    video_tracks.clear()
    audio_levels.clear()
    
    logger.info("Server shutdown complete")


def create_app():
    """Create and configure the web application"""
    app = web.Application()
    
    # Add routes
    app.router.add_post('/offer', offer_handler)
    app.router.add_get('/status', status_handler)
    app.router.add_get('/debug', debug_handler)
    app.router.add_get('/viewer', viewer_handler)
    app.router.add_get('/friends', friends_handler)
    app.router.add_get('/video-feed', video_feed_handler)
    app.router.add_get('/video-ws', video_websocket_handler)
    app.router.add_get('/', index_handler)
    app.router.add_static('/', path='static', name='static')
    
    # Add shutdown handler
    app.on_shutdown.append(on_shutdown)
    
    return app


async def main():
    """Main function to run the server"""
    logger.info("Starting WebRTC Conference Server...")
    
    app = create_app()
    
    # Start the web server
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 9000)
    await site.start()
    
    logger.info("Server running on http://0.0.0.0:9000")
    logger.info("Access the client at: http://YOUR_IP:9000")
    
    # Keep the server running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")