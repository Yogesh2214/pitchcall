#!/usr/bin/env python3
"""
Simple script to create an icon for the Pitch Recorder application
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    # Create a 32x32 icon
    size = 32
    icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    
    # Draw a microphone icon
    # Microphone body
    draw.ellipse([8, 12, 24, 28], fill=(52, 152, 219), outline=(41, 128, 185), width=2)
    
    # Microphone stand
    draw.rectangle([14, 8, 18, 12], fill=(52, 152, 219))
    
    # Sound waves
    for i in range(3):
        x = 26 + i * 4
        y = 16 + i * 2
        draw.arc([x, y, x + 4, y + 4], 0, 180, fill=(52, 152, 219), width=1)
    
    # Save as ICO
    icon.save('icon.ico', format='ICO')
    print("Icon created successfully: icon.ico")
    
except ImportError:
    print("PIL not available. Creating a simple text-based icon...")
    
    # Create a simple text file as fallback
    with open('icon.txt', 'w') as f:
        f.write("🎤")
    print("Created icon.txt as fallback")
    
except Exception as e:
    print(f"Error creating icon: {e}") 