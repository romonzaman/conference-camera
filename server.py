#!/usr/bin/env python3
"""
WebRTC Conference Server with Active Speaker Detection
Handles multiple client connections and switches video to active speaker
"""

import asyncio
import json
import logging
import time
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
        self.blank_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
    
    def set_active_speaker(self, speaker_id: str):
        """Switch to active speaker's video track"""
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
                logger.debug(f"Received frame from active speaker: {self.current_track}")
                return frame
            except Exception as e:
                logger.error(f"Error receiving frame from active speaker: {e}")
                self.current_track = None
        
        # Return blank frame if no active speaker
        logger.debug("No active speaker, returning blank frame")
        return self.blank_frame




def force_codec(pc, sender, forced_codec):
    """Force specific codec for better compatibility"""
    kind = forced_codec.split('/')[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences([codec for codec in codecs if codec.mimeType == forced_codec])


async def cleanup_client(client_id: str):
    """Clean up resources for disconnected client"""
    global active_speaker_id
    
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
                
                logger.debug(f"Audio level for {client_id}: {audio_level}")
                
            except Exception as e:
                logger.error(f"Error processing audio frame for {client_id}: {e}")
                break
                
    except Exception as e:
        logger.error(f"Audio monitoring ended for {client_id}: {e}")


async def detect_active_speaker():
    """Detect active speaker based on audio levels"""
    global active_speaker_id
    
    current_time = time.time()
    valid_speakers = {}
    
    # Find speakers with recent audio (within TIME_WINDOW)
    for client_id, audio_data in audio_levels.items():
        if current_time - audio_data['timestamp'] <= TIME_WINDOW:
            valid_speakers[client_id] = audio_data['level']
    
    if not valid_speakers:
        if active_speaker_id:
            active_speaker_id = None
            logger.info("No active speakers detected")
        return
    
    # Find speaker with highest audio level above threshold
    max_level = 0
    new_active_speaker = None
    
    for client_id, level in valid_speakers.items():
        if level > AUDIO_THRESHOLD and level > max_level:
            max_level = level
            new_active_speaker = client_id
    
    # Update active speaker if changed
    if new_active_speaker != active_speaker_id:
        active_speaker_id = new_active_speaker
        if active_speaker_id:
            logger.info(f"Active speaker changed to: {active_speaker_id} (level: {max_level})")
        else:
            logger.info("No speaker above threshold")


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
                await cleanup_client(client_id)
                pcs.discard(pc)
        
        # Handle incoming tracks
        @pc.on("track")
        def on_track(track):
            logger.info(f"Track received from {client_id}: {track.kind}")
            
            if track.kind == "video":
                # Store video track
                video_tracks[client_id] = track
                logger.info(f"Video track stored for {client_id}. Total video tracks: {len(video_tracks)}")
                
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