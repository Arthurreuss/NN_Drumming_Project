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

    def create_midi(self, drumroll_matrix, output_path, tempo=100):
        """
        Convert a (T,9) drumroll matrix with 0–1 values into a MIDI drum track.
        """
        cfg = load_config()
        if drumroll_matrix.ndim != 2 or drumroll_matrix.shape[1] != 9:
            raise ValueError("Expected matrix shape (T,9)")

        # Create MIDI object
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)

        # Determine timing
        fs = self._quantization
        step_duration = 60.0 / tempo / fs

        # Iterate over time steps and instruments
        for t in range(drumroll_matrix.shape[0]):  # timesteps
            for i, name in enumerate(cfg["dataset"]["pitch_groups"].keys()):
                if drumroll_matrix[t, i] > 0:
                    for pitch in cfg["dataset"]["pitch_groups"][name]:
                        note = pretty_midi.Note(
                            velocity=int(drumroll_matrix[t, i] * 127),
                            pitch=pitch,
                            start=t * step_duration,
                            end=(t + 1) * step_duration,
                        )
                        drum_instrument.notes.append(note)

        pm.instruments.append(drum_instrument)
        pm.write(output_path)
        print(f"[MIDI] Saved drum track to {output_path}")
