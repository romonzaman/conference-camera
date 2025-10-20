# 🎥 WebRTC Conference System

A smart conference camera system that automatically switches video to the active speaker using WebRTC technology. Perfect for remote meetings, online classes, or any scenario where you need to automatically focus on whoever is speaking.

## ✨ Features

### 🎯 Core Functionality
- **Multi-client support**: Multiple mobile devices can connect simultaneously
- **Active speaker detection**: Automatically detects who is speaking using audio level analysis
- **Smart video switching**: Switches the output video feed to the active speaker in real-time
- **WebRTC technology**: Low-latency, peer-to-peer communication
- **Mobile-optimized**: Responsive web interface designed for mobile browsers

### 🎨 User Interface
- **Real-time audio visualization**: Visual feedback showing audio levels
- **User identification**: Name and channel selection with persistent preferences
- **Mute/unmute controls**: Easy audio control with visual feedback
- **Full-screen video mode**: Immersive viewing experience when connected
- **Server viewer**: Dedicated page to view the server's output

### 🔧 Technical Features
- **Persistent preferences**: Remembers user name, channel, and mute status across sessions
- **Real-time status monitoring**: Live updates of connected clients and active speakers
- **Debug endpoints**: Built-in debugging tools for troubleshooting
- **Cross-platform compatibility**: Works on iOS, Android, and desktop browsers

## 🏗️ Architecture

```
Mobile Browsers → WebRTC → Python Server (aiortc) → Active Speaker Detection → Video Switching → Output
```

### System Flow
1. **Client Connection**: Mobile devices connect via WebRTC to the Python server
2. **Media Streaming**: Video and audio streams are sent to the server
3. **Audio Analysis**: Server analyzes audio levels to detect active speakers
4. **Video Switching**: Server automatically switches output to the active speaker
5. **Real-time Output**: Server streams the switched video to all connected clients

## 📁 Project Structure

```
conference-system/
├── server.py              # Python WebRTC server with active speaker detection
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── test_system.py        # System testing script
└── static/
    ├── index.html        # Mobile web client (main interface)
    ├── viewer.html       # Server video viewer (Tailwind CSS)
    └── friends.html      # Client list viewer (Tailwind CSS)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Modern web browser with WebRTC support
- Camera and microphone access

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd conference-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   python3 server.py
   ```

4. **Access the system**
   - Main client: `http://YOUR_IP:9000`
   - Server viewer: `http://YOUR_IP:9000/viewer`
   - Friends list: `http://YOUR_IP:9000/friends`

## 📱 Usage

### For Participants
1. Open your mobile browser and go to `http://YOUR_IP:9000`
2. Enter your name and select a channel (1-20)
3. Click "Join Conference" and grant camera/microphone permissions
4. Speak to see your audio level visualization
5. The server will automatically switch to whoever is speaking

### For Viewers
1. Open `http://YOUR_IP:9000/viewer` to see the server's output
2. Open `http://YOUR_IP:9000/friends` to see all connected participants
3. Watch as the video automatically switches to active speakers

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main client interface |
| `/viewer` | GET | Server video viewer |
| `/friends` | GET | Client list viewer |
| `/offer` | POST | WebRTC offer handler |
| `/status` | GET | System status (JSON) |
| `/debug` | GET | Debug information (JSON) |

## ⚙️ Configuration

You can modify these settings in `server.py`:

```python
AUDIO_THRESHOLD = 500      # Audio level threshold for speech detection
TIME_WINDOW = 1.0          # Time window for audio freshness (seconds)
VIDEO_WIDTH = 640          # Output video width
VIDEO_HEIGHT = 480         # Output video height
PORT = 9000                # Server port
```

## 🔧 Technical Details

### Backend (Python)
- **aiohttp**: Asynchronous web server framework
- **aiortc**: WebRTC implementation for Python
- **OpenCV**: Video processing and frame manipulation
- **NumPy**: Audio level calculations and signal processing
- **asyncio**: Asynchronous programming for concurrent connections

### Frontend (HTML/CSS/JS)
- **WebRTC API**: Peer connections and media streams
- **Web Audio API**: Real-time audio level visualization
- **Tailwind CSS**: Modern, responsive styling
- **LocalStorage**: Persistent user preferences
- **Responsive Design**: Mobile-first approach

### Active Speaker Detection Algorithm
1. **Audio Level Calculation**: RMS (Root Mean Square) of incoming audio
2. **Threshold Filtering**: Only speakers above 500 audio level are considered
3. **Time Window**: Only audio from the last 1 second is considered fresh
4. **Speaker Selection**: Highest audio level above threshold becomes active speaker
5. **Video Switching**: Server switches output to active speaker's video stream

## 🌐 Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ Full | Recommended |
| Safari | ✅ Full | iOS 11+ |
| Firefox | ✅ Full | All versions |
| Edge | ✅ Full | All versions |

## 🔒 Security Notes

- This is a development/demo system
- For production use, add HTTPS and authentication
- Consider implementing rate limiting and connection limits
- Ensure proper firewall configuration

## 🐛 Troubleshooting

### Common Issues

**Client can't connect:**
- Check if server is running on correct port
- Verify IP address is accessible from client device
- Ensure firewall allows connections on port 9000

**No video switching:**
- Check if multiple clients are connected
- Verify audio levels in debug endpoint (`/debug`)
- Ensure clients are speaking above threshold (500)

**Audio not detected:**
- Check microphone permissions in browser
- Verify audio levels in browser console
- Try speaking louder or adjusting threshold

### Debug Tools
- **Status endpoint**: `http://YOUR_IP:9000/status`
- **Debug endpoint**: `http://YOUR_IP:9000/debug`
- **Server logs**: Check console output for detailed information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [aiortc](https://github.com/aiortc/aiortc) for WebRTC implementation
- [aiohttp](https://github.com/aio-libs/aiohttp) for async web framework
- [Tailwind CSS](https://tailwindcss.com/) for styling framework
- WebRTC community for standards and documentation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review server logs for error messages
3. Test with different browsers/devices
4. Verify network connectivity
5. Open an issue on GitHub

---

**Made with ❤️ for seamless remote collaboration**

## Installation

1. **Install Python 3.9+** (if not already installed)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```bash
   python3 server.py
   ```

## Usage

1. **Start the server**:
   ```bash
   python3 server.py
   ```
   The server will start on `http://0.0.0.0:9000`

2. **Find your computer's IP address**:
   ```bash
   # On macOS/Linux:
   ifconfig | grep "inet "
   
   # On Windows:
   ipconfig
   ```

3. **Access the client**:
   - Open a mobile browser (Chrome, Safari, Firefox)
   - Navigate to `http://YOUR_IP:9000`
   - Click "Start Camera" and grant permissions
   - Multiple devices can connect to the same URL

4. **Test the system**:
   - Connect multiple devices
   - Speak on different devices
   - Watch the video automatically switch to the active speaker
   - Check the audio level visualization

## API Endpoints

- `GET /` - Serve the main web client
- `POST /offer` - Handle WebRTC offer from clients
- `GET /status` - Return system status (JSON)

## Configuration

You can modify these settings in `server.py`:

- `AUDIO_THRESHOLD = 500` - Audio level threshold for speech detection
- `TIME_WINDOW = 1.0` - Time window for audio freshness (seconds)
- `VIDEO_WIDTH = 640` - Output video width
- `VIDEO_HEIGHT = 480` - Output video height
- `PORT = 9000` - Server port

## Technical Details

### Backend (Python)
- **aiohttp**: Web server framework
- **aiortc**: WebRTC implementation
- **OpenCV**: Video processing
- **NumPy**: Audio level calculations
- **asyncio**: Asynchronous programming

### Frontend (HTML/CSS/JS)
- **WebRTC API**: Peer connections and media streams
- **Web Audio API**: Audio level visualization
- **Responsive design**: Mobile-first approach
- **Modern CSS**: Gradients, animations, flexbox

### Active Speaker Detection
- Calculates RMS (Root Mean Square) of audio levels
- Tracks audio levels per client with timestamps
- Considers only recent audio (within 1 second)
- Switches to speaker with highest level above threshold

## Troubleshooting

### Common Issues

1. **Camera not working**:
   - Ensure HTTPS or localhost (WebRTC requires secure context)
   - Check browser permissions
   - Try different browser

2. **Connection failed**:
   - Check firewall settings
   - Verify IP address is correct
   - Ensure server is running

3. **No audio detection**:
   - Check microphone permissions
   - Speak louder (increase threshold if needed)
   - Check browser audio settings

4. **Video not switching**:
   - Ensure multiple clients are connected
   - Check audio levels in browser console
   - Verify server logs for errors

### Debug Mode

Enable debug logging by modifying `server.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Browser Compatibility

- **Chrome**: Full support
- **Safari**: Full support (iOS 11+)
- **Firefox**: Full support
- **Edge**: Full support

## Security Notes

- This is a development/demo system
- For production use, add authentication and HTTPS
- Consider rate limiting and connection limits
- Implement proper error handling and logging

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs for errors
3. Test with different browsers/devices
4. Verify network connectivity