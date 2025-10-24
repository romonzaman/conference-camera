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
from settings import get_rtc_configuration, SERVER_HOST, SERVER_PORT, DEFAULT_LOG_LEVEL, MAX_CLIENTS, ACTIVE_SPEAKER_TIMEOUT, AUDIO_LEVEL_THRESHOLD, AUDIO_MONITORING_INTERVAL

# Configure logging
logging.basicConfig(level=getattr(logging, DEFAULT_LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

# Dynamic log level control
current_log_level = logging.INFO

# Set initial log levels for all relevant loggers
def initialize_log_levels():
    """Initialize log levels for all relevant loggers"""
    loggers_to_update = [
        'aiohttp.access',  # HTTP access logs
        'aiohttp',         # General aiohttp logs
        'aiortc',          # WebRTC logs
        'aioice',          # ICE/TURN logs
        'asyncio',         # AsyncIO logs
        'urllib3',         # HTTP client logs
        '__main__'         # Our main logger
    ]
    
    for logger_name in loggers_to_update:
        log = logging.getLogger(logger_name)
        log.setLevel(current_log_level)

# Initialize log levels
initialize_log_levels()

def set_log_level(level_name):
    """Set the global log level dynamically"""
    global current_log_level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'NOTICE': logging.WARNING  # Use WARNING for NOTICE to suppress INFO logs
    }
    if level_name in level_map:
        current_log_level = level_map[level_name]
        
        # List of all loggers to update
        loggers_to_update = [
            'aiohttp.access',  # HTTP access logs
            'aiohttp',         # General aiohttp logs
            'aiortc',          # WebRTC logs
            'aioice',          # ICE/TURN logs
            'asyncio',         # AsyncIO logs
            'urllib3',         # HTTP client logs
            '__main__'         # Our main logger
        ]
        
        # Update all loggers
        for logger_name in loggers_to_update:
            log = logging.getLogger(logger_name)
            log.setLevel(current_log_level)
        
        # Special handling for NOTICE level - completely disable access logs
        if level_name == 'NOTICE':
            aiohttp_access_logger = logging.getLogger('aiohttp.access')
            aiohttp_access_logger.disabled = True
            aiohttp_access_logger.setLevel(logging.CRITICAL)
        else:
            # Re-enable access logs for other levels
            aiohttp_access_logger = logging.getLogger('aiohttp.access')
            aiohttp_access_logger.disabled = False
            aiohttp_access_logger.setLevel(current_log_level)
        
        # Update root logger
        logging.getLogger().setLevel(current_log_level)
        
        # Log the change
        logger.info(f"Log level changed to {level_name} (level: {current_log_level})")
        
        return True
    return False

def smart_log(level, message, frequency_class="normal"):
    """
    Smart logging based on frequency classification
    - DEBUG: Very frequent logs (frame processing, audio levels)
    - INFO: Normal frequency logs (connections, state changes)
    - NOTICE: Important but less frequent logs (errors, major state changes)
    """
    if frequency_class == "high_frequency" and current_log_level > logging.DEBUG:
        return  # Skip high frequency logs unless DEBUG level
    elif frequency_class == "normal" and current_log_level > logging.INFO:
        return  # Skip normal logs unless INFO or DEBUG level
    elif frequency_class == "important" and current_log_level > logging.INFO:
        return  # Skip important logs unless INFO or DEBUG level
    
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)

# Global state management
pcs: Set[RTCPeerConnection] = set()
video_tracks: Dict[str, VideoStreamTrack] = {}
audio_tracks: Dict[str, AudioStreamTrack] = {}  # Store audio tracks
audio_levels: Dict[str, Dict] = {}
muted_clients: Set[str] = set()  # Track muted clients
active_speaker_id: Optional[str] = None
server_video_track = None
server_audio_track = None  # Audio track for viewers
client_info: Dict[str, Dict] = {}  # Store client name and channel info
viewer_websockets: Set[web.WebSocketResponse] = set()  # WebSocket connections for viewers

# Speaker detection state
last_speaker_change_time = 0
speaker_stability_delay = 0.5  # Reduced to 0.5 seconds for faster switching

# Real-time video streaming configuration
last_frame_time = 0
frame_interval = 1.0 / 30  # Increased to 30 FPS for real-time streaming
max_buffer_time = 2.0  # Maximum 2 seconds of buffering

# Configuration
AUDIO_THRESHOLD = AUDIO_LEVEL_THRESHOLD  # Audio level threshold from settings
TIME_WINDOW = 0.5  # Time window for audio detection (keep as is for now)
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# RTC Configuration from settings
def get_rtc_config():
    """Get RTC configuration from settings"""
    config = get_rtc_configuration()
    ice_servers = []
    
    for server in config["iceServers"]:
        if "username" in server and "credential" in server:
            # TURN server with credentials
            ice_servers.append(RTCIceServer(
                urls=server["urls"],
                username=server["username"],
                credential=server["credential"]
            ))
        else:
            # STUN server
            ice_servers.append(RTCIceServer(urls=server["urls"]))
    
    return RTCConfiguration(iceServers=ice_servers)


class AudioTransformTrack(AudioStreamTrack):
    """
    Audio track that switches between different client audio streams
    based on active speaker detection
    """
    
    def __init__(self):
        super().__init__()
        self.current_track = None
    
    def set_active_speaker(self, speaker_id: str):
        """Switch to active speaker's audio track"""
        smart_log("debug", f"Setting active speaker audio: {speaker_id}, available tracks: {list(audio_tracks.keys())}", "high_frequency")
        if speaker_id and speaker_id in audio_tracks:
            if self.current_track != audio_tracks[speaker_id]:
                self.current_track = audio_tracks[speaker_id]
                logger.info(f"Switched to speaker audio: {speaker_id}")
        else:
            if self.current_track is not None:
                self.current_track = None
                if speaker_id:
                    logger.info(f"Speaker {speaker_id} not found in audio tracks")
                else:
                    logger.info("No active speaker audio")
    
    async def recv(self):
        """Receive next audio frame with real-time processing and mute control"""
        if self.current_track:
            # Check if current speaker is muted
            if self.current_track and hasattr(self.current_track, 'client_id'):
                speaker_id = getattr(self.current_track, 'client_id', None)
                if speaker_id and speaker_id in muted_clients:
                    # Speaker is muted, return silence
                    logger.debug(f"Speaker {speaker_id} is muted, returning silent frame")
                    return self._create_silent_frame()
            
            try:
                # Use timeout for real-time audio
                frame = await asyncio.wait_for(
                    self.current_track.recv(), 
                    timeout=0.05  # 50ms timeout for real-time audio
                )
                
                smart_log("debug", f"Received audio frame from {getattr(self.current_track, 'client_id', 'unknown')}", "high_frequency")
                
                # Normalize audio frame to prevent resampler errors
                normalized_frame = self._normalize_audio_frame(frame)
                if normalized_frame is None:
                    logger.warning("Failed to normalize audio frame, returning silent frame")
                    return self._create_silent_frame()
                
                return normalized_frame
            except asyncio.TimeoutError:
                logger.debug("Audio frame timeout")
                self.current_track = None
            except Exception as e:
                logger.error(f"Error receiving audio frame from active speaker: {e}")
                # Reset audio processing state to recover from codec errors
                self._reset_audio_state()
                self.current_track = None
        
        # Return silence if no active speaker or speaker is muted
        return self._create_silent_frame()
    
    def _validate_audio_frame(self, frame):
        """Validate audio frame format to prevent resampler errors"""
        try:
            # Check if frame has valid audio data
            if not hasattr(frame, 'samples') or frame.samples <= 0:
                return False
            
            # Check sample rate
            if not hasattr(frame, 'sample_rate') or frame.sample_rate != 48000:
                return False
            
            # Check format
            if not hasattr(frame, 'format') or frame.format.name != 's16':
                return False
            
            # Check layout
            if not hasattr(frame, 'layout') or frame.layout.name != 'mono':
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Error validating audio frame: {e}")
            return False

    def _normalize_audio_frame(self, frame):
        """Normalize audio frame to ensure consistent format for Opus encoder"""
        try:
            # Convert to numpy array
            if hasattr(frame, 'to_ndarray'):
                audio_data = frame.to_ndarray()
            else:
                return None
            
            # Ensure it's the right shape and type
            if audio_data.ndim == 1:
                # Convert 1D to 2D (channels, samples)
                audio_data = audio_data.reshape(1, -1)
            elif audio_data.ndim == 2 and audio_data.shape[0] != 1:
                # Convert to mono if stereo
                audio_data = np.mean(audio_data, axis=0).reshape(1, -1)
            
            # Ensure it's int16
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            # Create new frame with consistent format
            normalized_frame = av.AudioFrame.from_ndarray(audio_data, format='s16', layout='mono')
            normalized_frame.sample_rate = 48000
            
            # Copy timestamp
            if hasattr(frame, 'pts') and frame.pts is not None:
                normalized_frame.pts = frame.pts
            else:
                if not hasattr(self, '_sample_count'):
                    self._sample_count = 0
                normalized_frame.pts = self._sample_count
                self._sample_count += audio_data.shape[1]
            
            return normalized_frame
        except Exception as e:
            logger.error(f"Error normalizing audio frame: {e}")
            return None

    def _create_silent_frame(self):
        """Create a silent audio frame with proper timestamps"""
        import av
        import numpy as np
        import time
        
        # Create a silent audio frame (48kHz, 20ms, mono)
        samples = 960  # 48kHz * 20ms = 960 samples
        silent_data = np.zeros((1, samples), dtype=np.int16)  # 2D array: (channels, samples)
        
        # Create audio frame
        frame = av.AudioFrame.from_ndarray(silent_data, format='s16', layout='mono')
        frame.sample_rate = 48000
        
        # Set proper timestamp to avoid Opus codec errors
        # Use sample-based timestamp (samples since start)
        if not hasattr(self, '_sample_count'):
            self._sample_count = 0
        frame.pts = self._sample_count
        self._sample_count += samples
        
        return frame

    def _reset_audio_state(self):
        """Reset audio processing state to recover from codec errors"""
        try:
            # Reset sample count to prevent timestamp issues
            if hasattr(self, '_sample_count'):
                self._sample_count = 0
            
            # Clear any cached audio data
            if hasattr(self, '_audio_buffer'):
                self._audio_buffer.clear()
            
            logger.debug("Audio state reset due to codec error")
        except Exception as e:
            logger.error(f"Error resetting audio state: {e}")


class VideoTransformTrack(VideoStreamTrack):
    """
    Real-time video track that switches between different client video streams
    based on active speaker detection with minimal buffering (max 2 seconds)
    """
    
    def __init__(self):
        super().__init__()
        self.current_track = None
        self.blank_frame = None
        self.frame_buffer = []  # Buffer for real-time frame management
        self.max_buffer_frames = 60  # Max 2 seconds at 30fps
        self.last_frame_time = 0
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
    
    def _validate_frame(self, frame):
        """Validate video frame before processing"""
        try:
            # Check if frame is valid
            if frame is None:
                return False
            
            # Check frame dimensions
            if not hasattr(frame, 'width') or not hasattr(frame, 'height'):
                return False
            
            if frame.width <= 0 or frame.height <= 0:
                return False
            
            # Check if frame has valid data
            if not hasattr(frame, 'to_ndarray'):
                return False
            
            # Basic validation without conversion to avoid errors
            try:
                # Just check if the method exists and can be called
                if hasattr(frame, 'format') and frame.format is not None:
                    return True
                else:
                    return False
            except Exception:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Frame validation error: {e}")
            return False

    def _normalize_video_frame(self, frame):
        """Normalize video frame to ensure consistent format for VP8 encoder"""
        try:
            # Convert to numpy array
            if hasattr(frame, 'to_ndarray'):
                video_data = frame.to_ndarray()
            else:
                return None
            
            # Handle different frame formats
            if video_data.ndim == 2:
                # Grayscale frame - convert to RGB
                logger.debug(f"Converting grayscale frame to RGB: {video_data.shape}")
                # Stack grayscale to create RGB
                video_data = np.stack([video_data, video_data, video_data], axis=2)
            elif video_data.ndim == 3:
                # Color frame - check format
                if video_data.shape[2] == 3:  # RGB format (HWC)
                    pass  # Keep as is
                elif video_data.shape[0] == 3:  # CHW format
                    # Convert CHW to HWC
                    video_data = np.transpose(video_data, (1, 2, 0))
                elif video_data.shape[2] == 4:  # RGBA format
                    # Convert RGBA to RGB
                    video_data = video_data[:, :, :3]
                else:
                    logger.warning(f"Unexpected 3D frame format: {video_data.shape}")
                    return None
            else:
                logger.warning(f"Unexpected frame dimensions: {video_data.ndim}, shape: {video_data.shape}")
                # Try to create a simple RGB frame as fallback
                try:
                    if video_data.ndim == 1:
                        # 1D array - try to reshape to square
                        size = int(np.sqrt(len(video_data)))
                        if size * size == len(video_data):
                            video_data = video_data.reshape(size, size)
                            video_data = np.stack([video_data, video_data, video_data], axis=2)
                        else:
                            return None
                    else:
                        return None
                except Exception:
                    return None
            
            # Ensure it's uint8
            if video_data.dtype != np.uint8:
                video_data = video_data.astype(np.uint8)
            
            # Validate final shape
            if video_data.shape[2] != 3:
                logger.error(f"Invalid final frame shape: {video_data.shape}")
                return None
            
            # Create new frame with consistent format
            normalized_frame = av.VideoFrame.from_ndarray(video_data, format='rgb24')
            normalized_frame.width = frame.width
            normalized_frame.height = frame.height
            
            # Copy timestamp
            if hasattr(frame, 'pts') and frame.pts is not None:
                normalized_frame.pts = frame.pts
            else:
                normalized_frame.pts = int(time.time() * 1000000)
            
            return normalized_frame
        except Exception as e:
            logger.error(f"Error normalizing video frame: {e}, frame shape: {getattr(video_data, 'shape', 'unknown') if 'video_data' in locals() else 'unknown'}")
            return None
    
    def set_active_speaker(self, speaker_id: str):
        """Switch to active speaker's video track"""
        smart_log("debug", f"Setting active speaker video: {speaker_id}", "high_frequency")
        smart_log("debug", f"Available video tracks: {list(video_tracks.keys())}", "high_frequency")
        
        if speaker_id and speaker_id in video_tracks:
            if self.current_track != video_tracks[speaker_id]:
                self.current_track = video_tracks[speaker_id]
                logger.debug(f"Switched to speaker video: {speaker_id}")
            else:
                logger.debug(f"Already using speaker video: {speaker_id}")
        else:
            if self.current_track is not None:
                self.current_track = None
                if speaker_id:
                    logger.info(f"Speaker {speaker_id} not found in video tracks")
                else:
                    logger.info("No active speaker, showing blank frame")
    
    async def recv(self):
        """Receive next video frame with real-time processing"""
        try:
            current_time = time.time()
            
            # Check if we need to drop old frames from buffer
            if current_time - self.last_frame_time > max_buffer_time:
                # Clear buffer if too much time has passed
                self.frame_buffer.clear()
                logger.debug("Cleared frame buffer due to timeout")
        
            # Try to get frame from current active speaker track
            if self.current_track:
                # Check if current speaker is muted - if so, skip processing
                if hasattr(self.current_track, 'client_id'):
                    speaker_id = getattr(self.current_track, 'client_id', None)
                    if speaker_id and speaker_id in muted_clients:
                        # Speaker is muted, return blank frame and skip buffering
                        logger.debug(f"Speaker {speaker_id} is muted, skipping video processing")
                        return self.blank_frame
            
                try:
                    # Use asyncio.wait_for to prevent blocking
                    frame = await asyncio.wait_for(
                        self.current_track.recv(), 
                        timeout=0.1  # 100ms timeout for real-time
                    )
                    
                    # Validate frame before processing
                    if not self._validate_frame(frame):
                        logger.warning("Invalid frame received, skipping")
                        return self.blank_frame
                    
                    # Use original frame directly - let the codec handle format conversion
                    # Just ensure the frame has proper timestamp
                    if frame.pts is None:
                        frame.pts = int(current_time * 1000000)
                    normalized_frame = frame
                    
                    # Update frame timestamp for proper codec handling
                    if normalized_frame.pts is None:
                        # Set timestamp if missing to avoid codec errors
                        normalized_frame.pts = int(current_time * 1000000)  # Convert to microseconds
                    self.last_frame_time = current_time
                    
                    # Manage buffer size - drop old frames if buffer is full
                    if len(self.frame_buffer) >= self.max_buffer_frames:
                        # Drop oldest frame
                        self.frame_buffer.pop(0)
                    
                    # Add new frame to buffer
                    self.frame_buffer.append(frame)
                
                    smart_log("debug", f"Received real-time frame: {frame.width}x{frame.height}", "high_frequency")
                    
                    # Broadcast frame to WebSocket viewers
                    try:
                        await self.broadcast_frame_to_viewers(frame)
                    except Exception as e:
                        logger.warning(f"Error broadcasting frame: {e}")
                    
                    return frame
                    
                except asyncio.TimeoutError:
                    logger.debug("Frame timeout - switching to fallback")
                    self.current_track = None
                except Exception as e:
                    logger.error(f"Error receiving frame from active speaker: {e}")
                    self.current_track = None
        
            # If no active speaker, try to get any available video track
            if not self.current_track and video_tracks:
                # Get the first available video track
                first_client = list(video_tracks.keys())[0]
                self.current_track = video_tracks[first_client]
                logger.debug(f"Using first available video track from {first_client}")
                try:
                    frame = await asyncio.wait_for(
                        self.current_track.recv(), 
                        timeout=0.1
                    )
                    # Ensure proper timestamp for codec handling
                    if frame.pts is None:
                        frame.pts = int(current_time * 1000000)  # Convert to microseconds
                    self.last_frame_time = current_time
                    
                    # Manage buffer
                    if len(self.frame_buffer) >= self.max_buffer_frames:
                        self.frame_buffer.pop(0)
                    self.frame_buffer.append(frame)
                    
                    smart_log("debug", f"Received fallback frame", "high_frequency")
                    
                    # Broadcast frame to WebSocket viewers
                    try:
                        await self.broadcast_frame_to_viewers(frame)
                    except Exception as e:
                        logger.warning(f"Error broadcasting fallback frame: {e}")
                
                    return frame
                    
                except asyncio.TimeoutError:
                    logger.debug("Fallback frame timeout")
                    self.current_track = None
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
        except Exception as e:
            logger.error(f"Error in video recv: {e}")
            # Return blank frame on any error
            if self.blank_frame:
                return self.blank_frame
            return None
    
    async def broadcast_frame_to_viewers(self, frame):
        """Broadcast video frame to WebSocket viewers with real-time optimization"""
        global last_frame_time
        
        if not viewer_websockets:
            return
        
        # More frequent rate limiting for real-time streaming
        current_time = time.time()
        if current_time - last_frame_time < (1.0 / 30):  # 30 FPS max
            return
        
        last_frame_time = current_time
        
        try:
            # Convert frame to numpy array
            frame_array = frame.to_ndarray(format="bgr24")
            
            # Resize frame to reduce bandwidth (smaller size for real-time)
            height, width = frame_array.shape[:2]
            small_frame = cv2.resize(frame_array, (width//3, height//3))  # Smaller for faster transmission
            
            # Convert to JPEG with optimized settings for real-time
            _, buffer = cv2.imencode('.jpg', small_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 30,  # Lower quality for real-time
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Progressive JPEG for faster loading
            ])
            
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Send to all connected viewers
            message = json.dumps({
                'type': 'video_frame',
                'data': frame_data,
                'timestamp': current_time
            })
            
            # Send to all viewers concurrently with timeout
            disconnected_viewers = set()
            for ws in viewer_websockets:
                try:
                    await asyncio.wait_for(ws.send_str(message), timeout=0.1)  # 100ms timeout
                except asyncio.TimeoutError:
                    logger.debug("WebSocket send timeout")
                    disconnected_viewers.add(ws)
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


async def clear_muted_client_buffers(client_id):
    """Clear all buffers and stop processing for a muted client"""
    try:
        logger.info(f"Clearing buffers for muted client: {client_id}")
        
        # Clear video buffer for this client
        if server_video_track and hasattr(server_video_track, 'frame_buffer'):
            # Clear all frames from this client
            original_count = len(server_video_track.frame_buffer)
            server_video_track.frame_buffer = [
                frame for frame in server_video_track.frame_buffer 
                if not (hasattr(frame, 'client_id') and frame.client_id == client_id)
            ]
            cleared_count = original_count - len(server_video_track.frame_buffer)
            if cleared_count > 0:
                logger.info(f"Cleared {cleared_count} video frames for muted client: {client_id}")
        
        # If this client was the active speaker, clear their tracks
        if server_video_track and hasattr(server_video_track, 'current_track'):
            if (hasattr(server_video_track.current_track, 'client_id') and 
                server_video_track.current_track.client_id == client_id):
                server_video_track.current_track = None
                logger.info(f"Cleared active speaker video for muted client: {client_id}")
        
        if server_audio_track and hasattr(server_audio_track, 'current_track'):
            if (hasattr(server_audio_track.current_track, 'client_id') and 
                server_audio_track.current_track.client_id == client_id):
                server_audio_track.current_track = None
                logger.info(f"Cleared active speaker audio for muted client: {client_id}")
        
        # Clear any remaining audio levels
        if client_id in audio_levels:
            del audio_levels[client_id]
            logger.info(f"Cleared audio levels for muted client: {client_id}")
        
        logger.info(f"Buffer clearing completed for muted client: {client_id}")
        
    except Exception as e:
        logger.error(f"Error clearing buffers for muted client {client_id}: {e}")


async def cleanup_client(client_id: str):
    """Clean up resources for disconnected client"""
    global active_speaker_id
    
    try:
        # Remove from video tracks
        if client_id in video_tracks:
            del video_tracks[client_id]
        
        # Remove from audio tracks
        if client_id in audio_tracks:
            del audio_tracks[client_id]
        
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
    """Monitor audio track and calculate levels (only for non-muted clients)"""
    try:
        while True:
            # Check if client is muted - if so, skip processing
            if client_id in muted_clients:
                logger.debug(f"Client {client_id} is muted, skipping audio processing")
                await asyncio.sleep(AUDIO_MONITORING_INTERVAL)  # Wait a bit before checking again
                continue
                
            try:
                frame = await track.recv()
                
                # Convert audio frame to numpy array
                audio_data = frame.to_ndarray()
                
                # Calculate RMS (Root Mean Square) for audio level
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)  # Convert to mono
                
                rms = np.sqrt(np.mean(audio_data**2))
                audio_level = int(rms * 10000)  # Increased scale for better visibility
                
                # Update audio levels with timestamp
                current_time = time.time()
                audio_levels[client_id] = {
                    'level': audio_level,
                    'timestamp': current_time
                }
                
                # Only log audio levels occasionally to avoid spam
                if audio_level % 100000 < 1000:  # Log roughly every 100th update
                    smart_log("info", f"Audio level for {client_id}: {audio_level}", "high_frequency")
                
                # Check for mute (very low audio level)
                if audio_level < 10:  # Very low threshold for mute detection
                    smart_log("debug", f"Client {client_id} appears muted (level: {audio_level})", "high_frequency")
                
                smart_log("debug", f"Audio level for {client_id}: {audio_level}", "high_frequency")
                
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
    """Detect active speaker based on audio levels with stability logic - only considers unmuted clients"""
    global active_speaker_id, last_speaker_change_time
    
    current_time = time.time()
    valid_speakers = {}
    
    # Find speakers with recent audio (within TIME_WINDOW) - ONLY UNMUTED CLIENTS
    for client_id, audio_data in audio_levels.items():
        # Only consider unmuted clients
        if client_id not in muted_clients and current_time - audio_data['timestamp'] <= TIME_WINDOW:
            valid_speakers[client_id] = audio_data['level']
    
    logger.debug(f"Audio levels: {audio_levels}")
    logger.debug(f"Valid speakers (unmuted only): {valid_speakers}")
    logger.debug(f"Muted clients: {list(muted_clients)}")
    
    # Find speaker with highest audio level above threshold
    max_level = 0
    new_active_speaker = None
    
    for client_id, level in valid_speakers.items():
        logger.debug(f"Client {client_id}: level={level}, threshold={AUDIO_THRESHOLD}")
        if level > AUDIO_THRESHOLD and level > max_level:
            max_level = level
            new_active_speaker = client_id
    
    # Simplified logic: switch based on unmute status and audio levels
    should_switch = False
    
    # If current speaker is muted, clear them immediately
    if active_speaker_id and active_speaker_id in muted_clients:
        should_switch = True
        new_active_speaker = None
        logger.debug(f"Current speaker {active_speaker_id} is muted, clearing active speaker")
    
    # If we have a new unmuted speaker with audio, switch to them
    elif new_active_speaker and new_active_speaker not in muted_clients:
        should_switch = True
        logger.debug(f"Switching to unmuted speaker: {new_active_speaker} (level: {max_level})")
    
    # If no current speaker and we have an unmuted speaker, switch to them
    elif active_speaker_id is None and new_active_speaker:
        should_switch = True
        logger.debug(f"No current speaker, switching to: {new_active_speaker} (level: {max_level})")
    
    if should_switch:
        old_speaker = active_speaker_id
        active_speaker_id = new_active_speaker
        last_speaker_change_time = current_time
        
        if active_speaker_id:
            logger.info(f"Active speaker changed to: {active_speaker_id} (level: {max_level})")
        else:
            logger.info("No active speaker - no unmuted speakers with audio")


async def offer_handler(request):
    """Handle WebRTC offer from client"""
    try:
        data = await request.json()
        offer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
        client_id = data.get('clientId', f'client_{int(time.time())}')
        user_name = data.get('userName', 'Unknown')
        channel = data.get('channel', 0)
        
        # Check if this is a viewer connection
        is_viewer = client_id.startswith('viewer_')
        
        # Store client information
        # ALL regular clients (non-viewers) are muted by default
        # Viewers are not muted by default (they don't have audio input)
        
        client_info[client_id] = {
            'name': user_name,
            'channel': channel,
            'connected_at': time.time(),
            'muted': not is_viewer  # All regular clients muted by default
        }
        
        # Add to muted clients set for all regular clients (non-viewers)
        if not is_viewer:
            muted_clients.add(client_id)
            logger.info(f"Client {client_id} connected as MUTED by default")
        else:
            logger.info(f"Viewer {client_id} connected (not muted - no audio input)")
        
        logger.info(f"Received offer from client: {client_id} (Name: {user_name}, Channel: {channel})")
        
        # Create peer connection
        pc = RTCPeerConnection(get_rtc_config())
        pcs.add(pc)
        
        # Create or get global video and audio transform tracks
        global server_video_track, server_audio_track
        if server_video_track is None:
            server_video_track = VideoTransformTrack()
        if server_audio_track is None:
            server_audio_track = AudioTransformTrack()
        
        # Handle connection state changes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.debug(f"Connection state for {client_id}: {pc.connectionState}")
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
            
            # Check if this is a viewer connection (not a regular client)
            is_viewer = client_id.startswith('viewer_')
            
            if track.kind == "video":
                if is_viewer:
                    # This is a viewer - they don't send video, they receive it
                    logger.info(f"Viewer {client_id} connected - will receive video stream")
                else:
                    # Store video track from regular client
                    video_tracks[client_id] = track
                    # Store client_id in track for mute checking
                    track.client_id = client_id
                    logger.debug(f"Video track stored for {client_id}. Total video tracks: {len(video_tracks)}")
                    
                    # If this is the first video track, set it as active speaker for testing
                    if len(video_tracks) == 1:
                        global active_speaker_id
                        active_speaker_id = client_id
                        logger.info(f"Set first video track as active speaker: {client_id}")
                        # Also update the video transform track immediately
                        if server_video_track:
                            server_video_track.set_active_speaker(client_id)
                
            elif track.kind == "audio":
                if is_viewer:
                    # This is a viewer - they don't send audio, they receive it
                    logger.info(f"Viewer {client_id} connected - will receive audio stream")
                else:
                    # Store audio track from regular client
                    audio_tracks[client_id] = track
                    # Store client_id in track for mute checking
                    track.client_id = client_id
                    logger.info(f"Audio track stored for {client_id}. Total audio tracks: {len(audio_tracks)}")
                    
                    # Connect the incoming audio track to our monitoring track
                    @track.on("ended")
                    def on_ended():
                        logger.info(f"Audio track ended for {client_id}")
                    
                    # Start monitoring the audio track only if client is not muted
                    if client_id not in muted_clients:
                        asyncio.create_task(monitor_audio_track(track, client_id))
                        logger.info(f"Audio monitoring started for {client_id}")
                    else:
                        logger.info(f"Audio monitoring skipped for muted client {client_id}")
        
        # Check if this is a viewer connection
        is_viewer = client_id.startswith('viewer_')
        
        if is_viewer:
            # For viewers, ensure server tracks exist and add them
            if server_video_track is None:
                server_video_track = VideoTransformTrack()
                logger.info("Created server video track for viewer")
            if server_audio_track is None:
                server_audio_track = AudioTransformTrack()
                logger.info("Created server audio track for viewer")
            
            # Add the server's output tracks
            pc.addTrack(server_video_track)
            pc.addTrack(server_audio_track)
            logger.debug(f"Added server output tracks for viewer: {client_id}")
            logger.debug(f"Current video tracks available: {list(video_tracks.keys())}")
            logger.debug(f"Current audio tracks available: {list(audio_tracks.keys())}")
        else:
            # For regular clients, they will send their own tracks
            logger.info(f"Regular client connection: {client_id}")
        
        # Handle the offer
        await pc.setRemoteDescription(offer)
        
        # Create answer with real-time optimizations
        answer = await pc.createAnswer()
        
        # Optimize SDP for real-time streaming
        sdp = answer.sdp
        
        # Add real-time optimizations to SDP
        sdp_lines = sdp.split('\n')
        optimized_sdp = []
        
        for line in sdp_lines:
            optimized_sdp.append(line)
            # Add low latency optimizations
            if line.startswith('m=video'):
                optimized_sdp.append('a=x-google-start-bitrate:1000')  # Start with higher bitrate
                optimized_sdp.append('a=x-google-max-bitrate:2000')    # Max bitrate
                optimized_sdp.append('a=x-google-min-bitrate:100')     # Min bitrate
            elif line.startswith('a=fmtp:') and 'VP8' in line:
                # Optimize VP8 for low latency
                optimized_sdp.append('a=fmtp:96 max-fr=30;max-fs=8192')
        
        sdp = '\n'.join(optimized_sdp)
        
        # Create new answer with optimized SDP
        answer = RTCSessionDescription(sdp=sdp, type=answer.type)
        await pc.setLocalDescription(answer)
        
        # Force codec preferences for real-time streaming
        for transceiver in pc.getTransceivers():
            if transceiver.sender and transceiver.sender.track and transceiver.sender.track.kind == 'video':
                # Prefer VP8 for low latency
                try:
                    force_codec(pc, transceiver.sender, 'video/VP8')
                except Exception as e:
                    logger.debug(f"Could not force VP8 codec: {e}")
            elif transceiver.sender and transceiver.sender.track and transceiver.sender.track.kind == 'audio':
                # Prefer Opus for low latency
                try:
                    force_codec(pc, transceiver.sender, 'audio/opus')
                except Exception as e:
                    logger.debug(f"Could not force Opus codec: {e}")
        
        # Active speaker detection loop runs globally
        
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


async def active_speaker_detection_loop():
    """Background task for real-time active speaker detection"""
    global server_video_track, server_audio_track
    
    while True:
        try:
            await detect_active_speaker()
            
            # Only update tracks if they exist
            if server_video_track:
                server_video_track.set_active_speaker(active_speaker_id)
            if server_audio_track:
                server_audio_track.set_active_speaker(active_speaker_id)
                
            await asyncio.sleep(0.05)  # Check every 50ms for real-time detection
        except Exception as e:
            logger.error(f"Error in speaker detection loop: {e}")
            await asyncio.sleep(AUDIO_MONITORING_INTERVAL)  # Shorter error recovery time


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
            "muted_clients": list(muted_clients),  # Include muted clients
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


async def mute_handler(request):
    """Toggle mute status for a client"""
    global active_speaker_id, last_speaker_change_time
    
    try:
        data = await request.json()
        client_id = data.get('client_id')
        
        if not client_id:
            return web.Response(status=400, text="client_id required")
        
        # Check if this is a viewer - viewers cannot be muted
        if client_id.startswith('viewer_'):
            return web.Response(status=400, text="Viewers cannot be muted (no audio input)")
        
        if client_id in muted_clients:
            # Unmute client
            muted_clients.remove(client_id)
            logger.info(f"Unmuted client: {client_id}")
            
            # Start audio monitoring if client has audio track
            if client_id in audio_tracks:
                track = audio_tracks[client_id]
                asyncio.create_task(monitor_audio_track(track, client_id))
                logger.info(f"Started audio monitoring for unmuted client: {client_id}")
            
            # Automatically set unmuted client as active speaker
            active_speaker_id = client_id
            last_speaker_change_time = time.time()
            logger.info(f"Set {client_id} as active speaker (unmuted)")
            
            return web.Response(text="Client unmuted")
        else:
            # Mute client
            muted_clients.add(client_id)
            logger.info(f"Muted client: {client_id}")
            
            # Clear active speaker if they were the active speaker
            if active_speaker_id == client_id:
                active_speaker_id = None
                last_speaker_change_time = time.time()
                logger.info(f"Cleared active speaker (muted): {client_id}")
            
            # Remove from audio levels to stop speaker detection
            if client_id in audio_levels:
                del audio_levels[client_id]
                logger.info(f"Removed audio levels for muted client: {client_id}")
            
            # Clear all buffers and stop processing for muted client
            await clear_muted_client_buffers(client_id)
            
            return web.Response(text="Client muted")
            
    except Exception as e:
        logger.error(f"Error toggling mute: {e}")
        return web.Response(status=500, text="Internal Server Error")


async def kick_handler(request):
    """Kick a client from the conference"""
    try:
        data = await request.json()
        client_id = data.get('client_id')
        
        if not client_id:
            return web.Response(status=400, text="client_id required")
        
        # Find and close the peer connection
        pc_to_close = None
        for pc in pcs:
            if hasattr(pc, 'client_id') and pc.client_id == client_id:
                pc_to_close = pc
                break
        
        if pc_to_close:
            await pc_to_close.close()
            logger.info(f"Kicked client: {client_id}")
            return web.Response(text="Client kicked")
        else:
            return web.Response(status=404, text="Client not found")
            
    except Exception as e:
        logger.error(f"Error kicking client: {e}")
        return web.Response(status=500, text="Internal Server Error")

async def log_level_handler(request):
    """Change the server log level dynamically"""
    try:
        data = await request.json()
        level = data.get('level', 'INFO').upper()
        
        if set_log_level(level):
            return web.Response(text=f"Log level changed to {level}")
        else:
            return web.Response(status=400, text="Invalid log level. Use DEBUG, INFO, or NOTICE")
    except Exception as e:
        logger.error(f"Error changing log level: {e}")
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


async def code_handler(request):
    """Serve the QR code page"""
    try:
        with open('static/code.html', 'r') as f:
            content = f.read()
        return web.Response(content_type="text/html", text=content)
    except FileNotFoundError:
        return web.Response(
            status=404, 
            text="<h1>404 - File not found</h1><p>Please ensure static/code.html exists</p>"
        )


async def code_info_handler(request):
    """Provide base URL information for QR code generation"""
    try:
        # Get the base URL from the request
        scheme = request.headers.get('X-Forwarded-Proto', 'http')
        host = request.headers.get('Host', request.host)
        
        # Construct the base URL
        base_url = f"{scheme}://{host}"
        
        # Return the base URL as JSON
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "base_url": base_url,
                "timestamp": time.time()
            })
        )
        
    except Exception as e:
        logger.error(f"Error getting base URL: {e}")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": "Internal Server Error"})
        )


async def qr_code_handler(request):
    """Generate QR code image on server side"""
    try:
        import qrcode
        from io import BytesIO
        import base64
        
        # Get the base URL from the request
        scheme = request.headers.get('X-Forwarded-Proto', 'http')
        host = request.headers.get('Host', request.host)
        base_url = f"{scheme}://{host}"
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(base_url)
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Return as data URL
        return web.Response(
            content_type="text/plain",
            text=f"data:image/png;base64,{img_str}"
        )
        
    except ImportError:
        logger.error("qrcode library not installed. Install with: pip install qrcode[pil]")
        return web.Response(
            status=500,
            text="QR code generation not available. Please install qrcode library."
        )
    except Exception as e:
        logger.error(f"Error generating QR code: {e}")
        return web.Response(
            status=500,
            text="Error generating QR code"
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


async def rtc_config_handler(request):
    """Get RTC configuration for frontend"""
    try:
        config = get_rtc_configuration()
        return web.json_response(config)
    except Exception as e:
        logger.error(f"Error getting RTC config: {e}")
        return web.Response(status=500, text="Internal Server Error")


def create_app():
    """Create and configure the web application"""
    app = web.Application()
    
    # Add routes
    app.router.add_post('/offer', offer_handler)
    app.router.add_get('/status', status_handler)
    app.router.add_get('/debug', debug_handler)
    app.router.add_get('/viewer', viewer_handler)
    app.router.add_get('/friends', friends_handler)
    app.router.add_get('/code', code_handler)
    app.router.add_get('/code-info', code_info_handler)
    app.router.add_get('/qr-code', qr_code_handler)
    app.router.add_get('/video-feed', video_feed_handler)
    app.router.add_get('/rtc-config', rtc_config_handler)
    app.router.add_post('/mute', mute_handler)
    app.router.add_post('/kick', kick_handler)
    app.router.add_post('/log-level', log_level_handler)
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
    
    site = web.TCPSite(runner, SERVER_HOST, SERVER_PORT)
    await site.start()
    
    # Start the active speaker detection loop
    asyncio.create_task(active_speaker_detection_loop())
    
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