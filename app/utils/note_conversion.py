"""
Utility functions for converting between different note formats.
Handles MIDI, tokens, and musical note representations.
"""

from typing import List, Dict, Any, Optional, Tuple
import mido
from ..schemas.requests import SeedNote
from ..schemas.responses import NoteResponse


def parse_midi_to_notes(midi_bytes: bytes) -> List[NoteResponse]:
    """Parse MIDI bytes to note representation."""
    try:
        import io
        midi_file = mido.MidiFile(file=io.BytesIO(midi_bytes))
        notes = []
        current_time = 0.0

        for track in midi_file.tracks:
            for msg in track:
                current_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, 500000)

                if msg.type == 'note_on' and msg.velocity > 0:
                    # Find corresponding note_off
                    note_end_time = current_time
                    remaining_time = current_time

                    for future_msg in track[track.index(msg)+1:]:
                        remaining_time += mido.tick2second(future_msg.time, midi_file.ticks_per_beat, 500000)
                        if (future_msg.type == 'note_off' and future_msg.note == msg.note) or \
                           (future_msg.type == 'note_on' and future_msg.note == msg.note and future_msg.velocity == 0):
                            note_end_time = remaining_time
                            break

                    duration = note_end_time - current_time
                    if duration > 0:
                        note = NoteResponse(
                            pitch=msg.note,
                            start_time=current_time,
                            duration=duration,
                            velocity=msg.velocity
                        )
                        notes.append(note)

        return sorted(notes, key=lambda n: n.start_time)

    except Exception as e:
        raise ValueError(f"Failed to parse MIDI: {e}")


def notes_to_midi_bytes(notes: List[NoteResponse], tempo: float = 120.0,
                       time_signature: Tuple[int, int] = (4, 4)) -> bytes:
    """Convert notes to MIDI bytes."""
    try:
        # Create MIDI file
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))

        # Add time signature
        track.append(mido.MetaMessage('time_signature',
                                    numerator=time_signature[0],
                                    denominator=time_signature[1]))

        # Convert notes to MIDI events
        events = []

        for note in notes:
            # Note on event
            events.append({
                'time': note.start_time,
                'type': 'note_on',
                'note': note.pitch,
                'velocity': note.velocity
            })

            # Note off event
            events.append({
                'time': note.start_time + note.duration,
                'type': 'note_off',
                'note': note.pitch,
                'velocity': 0
            })

        # Sort events by time
        events.sort(key=lambda e: e['time'])

        # Convert to MIDI messages
        current_time = 0.0
        for event in events:
            delta_time = max(0, event['time'] - current_time)
            delta_ticks = mido.second2tick(delta_time, mid.ticks_per_beat, mido.bpm2tempo(tempo))

            if event['type'] == 'note_on':
                msg = mido.Message('note_on',
                                 note=event['note'],
                                 velocity=event['velocity'],
                                 time=int(delta_ticks))
            else:
                msg = mido.Message('note_off',
                                 note=event['note'],
                                 velocity=event['velocity'],
                                 time=int(delta_ticks))

            track.append(msg)
            current_time = event['time']

        # Save to bytes
        import io
        buffer = io.BytesIO()
        mid.save(file=buffer)
        return buffer.getvalue()

    except Exception as e:
        raise ValueError(f"Failed to create MIDI: {e}")


def tokens_to_notes(tokens: List[int], tokenizer, time_resolution: int = 16) -> List[NoteResponse]:
    """Convert token sequence to notes using tokenizer."""
    notes = []
    current_time = 0.0

    try:
        for token in tokens:
            token_info = tokenizer.vocabulary.decode_token(token)
            if not token_info:
                continue

            if token_info['type'] == 'melody_note':
                pitch = token_info['pitch']
                duration_16ths = token_info['duration']
                velocity_name = token_info['velocity']

                # Convert duration from 16th notes to beats
                duration_beats = duration_16ths / 4.0

                # Map velocity names to MIDI values
                velocity_map = {'SOFT': 50, 'MEDIUM': 80, 'LOUD': 110}
                velocity = velocity_map.get(velocity_name, 80)

                note = NoteResponse(
                    pitch=pitch,
                    start_time=current_time,
                    duration=duration_beats,
                    velocity=velocity
                )
                notes.append(note)
                current_time += duration_beats

            elif token_info['type'] == 'rest':
                duration_16ths = token_info['duration']
                duration_beats = duration_16ths / 4.0
                current_time += duration_beats

    except Exception as e:
        raise ValueError(f"Failed to convert tokens to notes: {e}")

    return notes


def notes_to_tokens(notes: List[SeedNote], tokenizer, note_format: str = "beats") -> List[int]:
    """Convert notes to tokens using tokenizer."""
    # This is a simplified implementation
    # In practice, you'd need to implement the reverse of your tokenization process
    tokens = []

    try:
        for note in notes:
            # Convert note to appropriate token representation
            # This depends on your specific tokenizer implementation

            # For now, this is a placeholder
            # You would implement the actual note-to-token conversion here
            pass

    except Exception as e:
        raise ValueError(f"Failed to convert notes to tokens: {e}")

    return tokens


def convert_time_format(notes: List[NoteResponse], from_format: str, to_format: str,
                       tempo: float = 120.0, ticks_per_beat: int = 480) -> List[NoteResponse]:
    """Convert note timing between different formats."""
    if from_format == to_format:
        return notes

    converted_notes = []

    for note in notes:
        new_note = NoteResponse(
            pitch=note.pitch,
            velocity=note.velocity,
            channel=note.channel,
            start_time=note.start_time,
            duration=note.duration
        )

        # Convert start_time and duration
        if from_format == "beats" and to_format == "seconds":
            beats_per_second = tempo / 60.0
            new_note.start_time = note.start_time / beats_per_second
            new_note.duration = note.duration / beats_per_second

        elif from_format == "seconds" and to_format == "beats":
            beats_per_second = tempo / 60.0
            new_note.start_time = note.start_time * beats_per_second
            new_note.duration = note.duration * beats_per_second

        elif from_format == "beats" and to_format == "ticks":
            new_note.start_time = note.start_time * ticks_per_beat
            new_note.duration = note.duration * ticks_per_beat

        elif from_format == "ticks" and to_format == "beats":
            new_note.start_time = note.start_time / ticks_per_beat
            new_note.duration = note.duration / ticks_per_beat

        elif from_format == "seconds" and to_format == "ticks":
            beats_per_second = tempo / 60.0
            beats_start = note.start_time * beats_per_second
            beats_duration = note.duration * beats_per_second
            new_note.start_time = beats_start * ticks_per_beat
            new_note.duration = beats_duration * ticks_per_beat

        elif from_format == "ticks" and to_format == "seconds":
            beats_per_second = tempo / 60.0
            beats_start = note.start_time / ticks_per_beat
            beats_duration = note.duration / ticks_per_beat
            new_note.start_time = beats_start / beats_per_second
            new_note.duration = beats_duration / beats_per_second

        converted_notes.append(new_note)

    return converted_notes