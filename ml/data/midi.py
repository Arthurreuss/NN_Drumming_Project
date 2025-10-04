import enum

import numpy as np
from mido import MidiFile
from unidecode import unidecode


class Midi:
    def __init__(self, quantization):
        self._song_path = None
        self._quantization = quantization
        self._pitch_dim = 128  # pitch dimension
        self._total_ticks = None  # length in MIDI ticks
        self._total_beats = None  # time steps in quantized drum matrix
        self.tracks = {}  # drum matrices for each track
        self._notes_on_temp = []
        self._unnamed_tracks = 0

    def _get_total_ticks(self):
        mid = MidiFile(self._song_path)
        num_ticks = 0
        for i, track in enumerate(mid.tracks):
            tick_counter = 0
            for message in track:
                tick_counter += float(message.time)
            num_ticks = max(num_ticks, tick_counter)
        self._total_ticks = num_ticks

    def _get_pitch_range(self):
        mid = MidiFile(self._song_path)
        min_pitch, max_pitch = 200, 0
        for i, track in enumerate(mid.tracks):
            for message in track:
                if message.type in ["note_on", "note_off"]:
                    pitch = message.note
                    min_pitch = min(min_pitch, pitch)
                    max_pitch = max(max_pitch, pitch)
        return min_pitch, max_pitch

    def _get_total_beats(self):
        mid = MidiFile(self._song_path)
        ticks_per_beat = mid.ticks_per_beat
        self._get_total_ticks()
        self._total_beats = int(
            (self._total_ticks / ticks_per_beat) * self._quantization
        )

    def _add_note_to_drum_matrix(self, note_off, track_name):
        pitch_off, _, beat_off = note_off
        for idx, (pitch_on, velocity_on, beat_on) in enumerate(self._notes_on_temp):
            if pitch_on == pitch_off:
                self.tracks[track_name]["drum_matrix"][
                    beat_on:beat_off, pitch_on
                ] = velocity_on

                del self._notes_on_temp[idx]
                return

    def read_file(self, song_path):
        self._song_path = song_path
        mid = MidiFile(self._song_path)
        ticks_per_beat = mid.ticks_per_beat
        self._get_total_beats()

        for idx_track, track in enumerate(mid.tracks):
            self._notes_on_temp = []
            name = unidecode(track.name).rstrip("\x00")
            if name == "":
                name = f"unnamed_{self._unnamed_tracks}"
                self._unnamed_tracks += 1
            self.tracks[name] = {
                "drum_matrix": np.zeros(
                    [self._total_beats, self._pitch_dim], dtype=np.int16
                ),
                "genre": None,  # todo
            }

            abs_ticks = 0
            for message in track:
                abs_ticks += message.time  # accumulate ticks
                beat = int(abs_ticks * self._quantization / ticks_per_beat)

                if message.type == "note_on":
                    pitch, velocity = message.note, message.velocity
                    if velocity > 0:
                        self._notes_on_temp.append((pitch, velocity, beat))
                    else:
                        self._add_note_to_drum_matrix((pitch, 0, beat), name)

                elif message.type == "note_off":
                    pitch, velocity = message.note, message.velocity
                    self._add_note_to_drum_matrix((pitch, velocity, beat), name)
            if np.count_nonzero(self.tracks[name]["drum_matrix"]) == 0:
                del self.tracks[name]
        return self.tracks

    def create_midi(self, drum_matrix, output_path, tempo=100):
        # todo
        pass
