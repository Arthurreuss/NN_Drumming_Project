import logging

import numpy as np
import pretty_midi
from mido import MidiFile
from unidecode import unidecode

from ml.utils.cfg import load_config


class Midi:
    def __init__(self, quantization):
        self._song_path = None
        self._quantization = quantization
        self._pitch_dim = 128  # pitch dimension
        self._total_ticks = None  # length in MIDI ticks
        self._total_timesteps = None  # time steps in quantized pianoroll matrix
        self.tracks = {}  # pianoroll matrices for each track
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

    def _get_total_timesteps(self):
        mid = MidiFile(self._song_path)
        ticks_per_beat = mid.ticks_per_beat
        self._get_total_ticks()
        self._total_timesteps = int(
            (self._total_ticks / ticks_per_beat) * self._quantization
        )

    def _add_note_to_pianoroll_matrix(self, note_off, track_name):
        pitch_off, _, timestep_off = note_off
        for idx, (pitch_on, velocity_on, timestep_on) in enumerate(self._notes_on_temp):
            if pitch_on == pitch_off:
                self.tracks[track_name][
                    timestep_on:timestep_off, pitch_on
                ] = velocity_on

                del self._notes_on_temp[idx]
                return

    def read_file(self, song_path):
        self._song_path = song_path
        mid = MidiFile(self._song_path)
        ticks_per_beat = mid.ticks_per_beat
        self._get_total_timesteps()

        for idx_track, track in enumerate(mid.tracks):
            self._notes_on_temp = []
            name = unidecode(track.name).rstrip("\x00")
            if name == "":
                name = f"unnamed_{self._unnamed_tracks}"
                self._unnamed_tracks += 1
            self.tracks[name] = np.zeros(
                [self._total_timesteps, self._pitch_dim], dtype=np.int16
            )

            abs_ticks = 0
            for message in track:
                abs_ticks += message.time  # accumulate ticks
                timestep = int(abs_ticks * self._quantization / ticks_per_beat)

                if message.type == "note_on":
                    pitch, velocity = message.note, message.velocity
                    if velocity > 0:
                        self._notes_on_temp.append((pitch, velocity, timestep))
                    else:
                        self._add_note_to_pianoroll_matrix((pitch, 0, timestep), name)

                elif message.type == "note_off":
                    pitch, velocity = message.note, message.velocity
                    self._add_note_to_pianoroll_matrix(
                        (pitch, velocity, timestep), name
                    )
            if np.count_nonzero(self.tracks[name]) == 0:
                del self.tracks[name]
        return self.tracks

    def create_midi(self, cfg, drumroll_matrix, output_path, tempo=100, threshold=0.3):
        """
        Convert a (T,9) drumroll matrix (values 0–1) into a clean, playable MIDI drum file.
        """
        pitch_groups = cfg["dataset_creation"]["pitch_groups"]

        if drumroll_matrix.ndim != 2 or drumroll_matrix.shape[1] != len(pitch_groups):
            raise ValueError(f"Expected matrix shape (T,{len(pitch_groups)})")

        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)

        # Correct step duration: 16 steps per bar = 4 beats per bar * 4 steps per beat
        step_duration = 60.0 / tempo / self._quantization

        drumroll_matrix = np.clip(drumroll_matrix, 0, 1)
        drumroll_matrix = (drumroll_matrix > threshold).astype(int)

        for i, name in enumerate(pitch_groups.keys()):
            pitches = pitch_groups[name]
            active = False
            note_start = 0

            for t in range(drumroll_matrix.shape[0]):
                val = drumroll_matrix[t, i]
                if val and not active:
                    active = True
                    note_start = t
                elif not val and active:
                    active = False
                    start_time = note_start * step_duration
                    end_time = t * step_duration
                    pitch = pitches[0]
                    note = pretty_midi.Note(
                        velocity=100, pitch=pitch, start=start_time, end=end_time
                    )
                    drum_instrument.notes.append(note)

            # Close last note if sequence ends while active
            if active:
                start_time = note_start * step_duration
                end_time = drumroll_matrix.shape[0] * step_duration
                for pitch in pitches:
                    note = pretty_midi.Note(
                        velocity=100, pitch=pitch, start=start_time, end=end_time
                    )
                    drum_instrument.notes.append(note)

        pm.instruments.append(drum_instrument)
        pm.write(output_path)
        logging.info(f"[MIDI] Saved cleaned drum track to {output_path}")
