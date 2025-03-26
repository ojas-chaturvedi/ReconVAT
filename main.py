"""
Usage: python main.py -i input.mp3 -o output.mid --model_type ReconVAT --device cpu
"""

import pickle
import os
import argparse
import numpy as np
import torch
from model import *
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf


def convert_mp3_to_16k_flac(input_mp3_path, output_flac_path):
    audio = AudioSegment.from_mp3(input_mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # mono 16kHz
    sf.write(output_flac_path, audio.get_array_of_samples(), 16000, format="FLAC")
    print(f"Converted: {input_mp3_path} → {output_flac_path}")


def transcribe2midi(
    data,
    model,
    model_type,
    onset_threshold=0.5,
    frame_threshold=0.5,
    save_path=None,
    reconstruction=True,
    onset=True,
    pseudo_onset=False,
    rule="rule2",
    VAT=False,
):
    for i in data:
        pred = model.transcribe(i)
        #         print(f"pred['onset2'] = {pred['onset2'].shape}")
        #         print(f"pred['frame2'] = {pred['frame2'].shape}")

        for key, value in pred.items():
            if key in ["frame", "onset", "frame2", "onset2"]:
                value.squeeze_(
                    0
                ).relu_()  # remove batch dim and remove make sure no negative values
            p_est, i_est = extract_notes_wo_velocity(
                pred["onset"],
                pred["frame"],
                onset_threshold,
                frame_threshold,
                rule=rule,
            )

        # print(f"p_ref = {p_ref}\n p_est = {p_est}")

        t_est, f_est = notes_to_frames(p_est, i_est, pred["frame"].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        # Converting time steps to seconds and midi number to frequency
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_est = t_est.astype(np.float64) * scaling
        f_est = [
            np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est
        ]

        # If save_path is a directory, generate a file name
        if os.path.isdir(save_path):
            midi_filename = (
                model_type + "-" + os.path.basename(i["path"]).replace(".flac", ".mid")
            )
            midi_path = os.path.join(save_path, midi_filename)
        else:
            midi_path = save_path

        print(f"midi_path = {midi_path}")
        save_midi(midi_path, p_est, i_est, [127] * len(p_est))


@ex.config
def config():
    device = "cuda:0"
    model_type = "ReconVAT"
    # instrument='string'


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a .mp3 file to .mid using a selected model."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input .mp3 file")
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output .mid file"
    )
    parser.add_argument(
        "--model_type",
        default="ReconVAT",
        choices=["ReconVAT", "baseline_Multi_Inst"],
        help="Model to use",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Device to run model on (e.g. "cuda:0" or "cpu")',
    )

    args = parser.parse_args()

    log = True
    mode = "imagewise"
    spec = "Mel"

    import tempfile
    import shutil

    tmp_input_dir = tempfile.mkdtemp()
    flac_filename = os.path.splitext(os.path.basename(args.input))[0] + ".flac"
    flac_path = os.path.join(tmp_input_dir, flac_filename)
    convert_mp3_to_16k_flac(args.input, flac_path)

    # Load audios from the Input files
    application_dataset = Application_Dataset(tmp_input_dir, device=args.device)

    if args.model_type == "ReconVAT":
        model = UNet(
            (2, 2),
            (2, 2),
            log=log,
            reconstruction=True,
            mode=mode,
            spec=spec,
            device=args.device,
        )
        weight_path = "Weight/String_MusicNet/Unet_R_VAT-XI=1e-06-eps=1.3-String_MusicNet-lr=0.001/weight.pt"
    elif args.model_type == "baseline_Multi_Inst":
        model = Semantic_Segmentation(
            torch.empty(1, 1, 640, N_BINS), 1, device=args.device
        )
        weight_path = "Weight/String_MusicNet/baseline_Multi_Inst/weight.pt"

    print(f"Loading model weight")
    model.load_state_dict(torch.load(weight_path, map_location=args.device))
    model.to(args.device)
    print(f"Loading done")
    print(f"Transcribing Music")
    transcribe2midi(
        tqdm(application_dataset),
        model,
        args.model_type,
        reconstruction=False,
        save_path=os.path.dirname(args.output),
    )

    base_flac_name = os.path.basename(flac_path).replace(".flac", "")
    generated_name = f"{args.model_type}-{base_flac_name}.mid"
    generated_path = os.path.join(os.path.dirname(args.output), generated_name)

    if os.path.exists(generated_path):
        os.rename(generated_path, args.output)
        print(f"Saved final MIDI to: {args.output}")
    else:
        print(f"Error: MIDI not found at {generated_path}")

    shutil.rmtree(tmp_input_dir)


if __name__ == "__main__":
    main()
