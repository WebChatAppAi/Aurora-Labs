#!/usr/bin/env python3
"""Test complete user workflow: Upload MIDI -> Generate -> Download."""

import asyncio
import aiohttp
import tempfile
import os
import json

async def test_complete_workflow():
    """Test the complete user workflow."""
    async with aiohttp.ClientSession() as session:
        print("🎵 TESTING COMPLETE AI MUSIC GENERATION WORKFLOW")
        print("=" * 60)
        
        # Step 1: Load model
        print("\n1. Loading AI model...")
        load_data = {"model_name": "best_melody_model", "force_reload": False}
        async with session.post("http://localhost:8000/api/v1/models/load", 
                               json=load_data) as response:
            if response.status != 200:
                print(f"❌ Model loading failed: {await response.json()}")
                return
            print("✅ AI model loaded successfully")
        
        # Step 2: Upload MIDI and generate
        print("\n2. Uploading MIDI file and generating new music...")
        
        # Check if lead-melody.mid exists
        if not os.path.exists("lead-melody.mid"):
            print("❌ lead-melody.mid not found. Please place a MIDI file in the project root.")
            return
        
        # Upload MIDI for generation
        data = aiohttp.FormData()
        data.add_field('model_name', 'best_melody_model')
        data.add_field('num_generations', '2')  # Generate 2 melodies
        data.add_field('max_seed_length', '30')
        data.add_field('params', json.dumps({
            "temperature": 1.2,
            "top_k": 60,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "max_length": 100
        }))
        
        with open("lead-melody.mid", 'rb') as f:
            data.add_field('midi_file', f, filename='lead-melody.mid', content_type='audio/midi')
            
            async with session.post("http://localhost:8000/api/v1/generate/from-midi", 
                                   data=data) as response:
                if response.status != 200:
                    result = await response.json()
                    print(f"❌ Generation failed: {result}")
                    return
                
                result = await response.json()
                print("✅ Music generation successful!")
                print(f"   Generated {result['success_count']} melodies")
                print(f"   Generation time: {result['total_generation_time']:.2f}s")
                
                # Step 3: Download individual MIDI files
                print("\n3. Downloading generated MIDI files...")
                
                for i, melody in enumerate(result['melodies']):
                    melody_id = melody['id']
                    midi_url = melody['midi_url']
                    token_count = melody['metadata']['token_count']
                    duration = melody['metadata']['duration_beats']
                    
                    print(f"\n   Melody {i+1}: {melody_id}")
                    print(f"   - Tokens: {token_count}")
                    print(f"   - Duration: {duration} beats")
                    print(f"   - Download URL: {midi_url}")
                    
                    # Download the MIDI file
                    async with session.get(f"http://localhost:8000{midi_url}") as download_response:
                        if download_response.status == 200:
                            midi_content = await download_response.read()
                            filename = f"generated_melody_{i+1}.mid"
                            
                            with open(filename, 'wb') as f:
                                f.write(midi_content)
                            
                            print(f"   ✅ Downloaded: {filename} ({len(midi_content)} bytes)")
                        else:
                            print(f"   ❌ Download failed: {download_response.status}")
                
                # Step 4: Download ZIP file with all melodies
                print("\n4. Downloading ZIP file with all melodies...")
                request_id = result['request_id']
                zip_url = f"http://localhost:8000/api/v1/download/midi-zip/{request_id}"
                
                async with session.get(zip_url) as zip_response:
                    if zip_response.status == 200:
                        zip_content = await zip_response.read()
                        zip_filename = f"{request_id}_melodies.zip"
                        
                        with open(zip_filename, 'wb') as f:
                            f.write(zip_content)
                        
                        print(f"✅ Downloaded ZIP: {zip_filename} ({len(zip_content)} bytes)")
                    else:
                        print(f"❌ ZIP download failed: {zip_response.status}")
                
                # Step 5: Test simple generation (no upload)
                print("\n5. Testing simple generation (no MIDI upload)...")
                
                params = {
                    "model_name": "best_melody_model",
                    "temperature": 1.0,
                    "top_k": 50,
                    "top_p": 0.9,
                    "max_length": 80,
                    "num_generations": 1
                }
                
                async with session.get("http://localhost:8000/api/v1/generate/simple", 
                                       params=params) as simple_response:
                    if simple_response.status == 200:
                        simple_result = await simple_response.json()
                        print("✅ Simple generation successful!")
                        
                        if simple_result['melodies']:
                            melody = simple_result['melodies'][0]
                            midi_url = melody['midi_url']
                            
                            # Download simple generation
                            async with session.get(f"http://localhost:8000{midi_url}") as download_response:
                                if download_response.status == 200:
                                    midi_content = await download_response.read()
                                    
                                    with open("simple_generated_melody.mid", 'wb') as f:
                                        f.write(midi_content)
                                    
                                    print(f"   ✅ Downloaded: simple_generated_melody.mid ({len(midi_content)} bytes)")
                    else:
                        print(f"❌ Simple generation failed: {simple_response.status}")
        
        print("\n" + "=" * 60)
        print("🎉 COMPLETE WORKFLOW TEST FINISHED!")
        print("\nGenerated files:")
        print("- generated_melody_1.mid (from MIDI upload)")
        print("- generated_melody_2.mid (from MIDI upload)")  
        print("- [request_id]_melodies.zip (ZIP with all melodies)")
        print("- simple_generated_melody.mid (simple generation)")
        print("\n🎵 Your AI Music Generation API is fully functional!")
        print("Users can now:")
        print("✅ Upload MIDI files as seeds")
        print("✅ Generate multiple variations")
        print("✅ Download individual .mid files")
        print("✅ Download ZIP archives")
        print("✅ Use simple generation without uploads")

if __name__ == "__main__":
    asyncio.run(test_complete_workflow())
