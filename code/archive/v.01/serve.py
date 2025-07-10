#!/usr/bin/env python3
"""
Simple HTTP server for CVE Forecast Dashboard
Run this script to serve the dashboard locally
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

def main():
    # Change to web directory
    os.chdir('web')
    
    # Set up server
    PORT = 8000
    
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers for local development
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            
            # Add cache-busting headers for data.json to prevent stale data
            if self.path.endswith('data.json'):
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
            
            super().end_headers()
    
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"CVE Forecast Dashboard server starting...")
        print(f"Dashboard available at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop the server")
        
        # Try to open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass
        
        # Start server
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
