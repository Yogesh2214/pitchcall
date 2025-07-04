# 🎤 PitchCall - Interactive Investor Call Recorder and Transcriber

This application helps you record audio from various sources (Zoom, Google Meet, regular calls), transcribe it using ChatGPT, and generate investor-ready notes from pitch calls with a modern, interactive interface.

## ✨ Features
- **🎙️ Real-time Audio Recording** with live visualization
- **📊 Audio Waveform Display** showing recording levels and volume
- **🎯 Interactive GUI** with modern styling and intuitive controls
- **⌨️ Keyboard Shortcuts** for quick access to common functions
- **📝 Smart Transcription** using OpenAI's Whisper API
- **📋 AI-Generated Notes** structured for investors
- **💾 Multiple Save Options** (transcript, notes, or both)
- **📋 Copy to Clipboard** functionality
- **🎨 Modern UI** with emojis and color-coded status indicators

## 🚀 Setup
1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. (Optional) Generate an application icon:
   ```bash
   python create_icon.py
   ```

## 🎮 Usage

### Running the Application
```bash
python main.py
```

### Interactive Controls
- **🎙️ Start/Stop Recording**: Click the record button or press `Ctrl+R`
- **📝 Transcribe**: Click transcribe button or press `Ctrl+T`
- **💾 Save**: Use `Ctrl+S` to save both transcript and notes
- **📋 Copy**: Use `Ctrl+C` to copy transcript or `Ctrl+N` to copy notes
- **⏹️ Stop Recording**: Press `Esc` to stop recording immediately
- **🗑️ Clear**: Clear all data and reset the interface

### Workflow
1. **Record**: Start recording your pitch call (supports any audio source)
2. **Visualize**: Watch the real-time audio waveform and volume levels
3. **Transcribe**: Let AI transcribe your audio with high accuracy
4. **Review**: Check the generated transcript and investor notes
5. **Save**: Save your work in multiple formats

## 🎨 UI Features
- **Real-time Audio Visualization**: See your recording levels live
- **Color-coded Status**: Green for success, red for errors, orange for warnings
- **Progress Indicators**: Visual feedback for all operations
- **Tabbed Interface**: Separate tabs for transcript and notes
- **Responsive Design**: Adapts to different window sizes
- **Modern Styling**: Clean, professional appearance

## ⌨️ Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Start/Stop Recording |
| `Ctrl+T` | Transcribe Audio |
| `Ctrl+S` | Save Both Files |
| `Ctrl+C` | Copy Transcript |
| `Ctrl+N` | Copy Notes |
| `Esc` | Stop Recording |

## 📋 Generated Notes Structure
The AI automatically generates structured investor notes including:
- **Key Points**: Main takeaways from the pitch
- **Investment Ask**: Funding requirements and terms
- **Market Opportunity**: Market size and growth potential
- **Competitive Advantage**: Unique selling propositions
- **Financial Highlights**: Revenue, growth, and metrics
- **Team Members**: Key personnel and their backgrounds
- **Experience**: Team's relevant experience and track record
- **Use of Funds**: How the investment will be allocated
- **Financial Projections**: Future revenue and growth forecasts
- **Exit Strategy**: Potential exit scenarios and timelines
- **Risks**: Key risks and mitigation strategies
- **Conclusion**: Summary and next steps

## 🔧 Technical Features
- **Large File Support**: Automatically splits files over 25MB
- **High-Quality Transcription**: Uses Whisper API with business context
- **Error Handling**: Graceful handling of API errors and file issues
- **Multi-threading**: Non-blocking UI during operations
- **File Management**: Automatic cleanup of temporary files

## 📦 Requirements
- Python 3.8+
- OpenAI API key
- Working microphone
- Internet connection for transcription
- matplotlib (for visualization)

## ⚠️ Important Notes
- Make sure you have permission to record calls
- Comply with local laws and regulations regarding audio recording
- The application requires an active internet connection for transcription
- Large audio files may take longer to process

## 🐛 Troubleshooting
- **No Audio**: Check your microphone permissions and settings
- **Transcription Errors**: Verify your OpenAI API key and internet connection
- **Large Files**: The app automatically handles files over 25MB by splitting them
- **Visualization Issues**: Ensure matplotlib is properly installed

## 📄 License
This project is for educational and business use. Please respect privacy and recording laws in your jurisdiction.

 