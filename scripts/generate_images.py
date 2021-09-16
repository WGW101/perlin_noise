import argparse
import os
import time
import random

import tqdm
import numpy as np
import imageio

from perlin_noise import fractal2d


def parse_args() -> argparse.Namespace:
      parser = argparse.ArgumentParser(add_help=False)
      parser.add_argument("--num-images", "-n", type=int, default=1)
      parser.add_argument("--out-tmpl", "-o", default=None)
      parser.add_argument("--width", "-w", type=int, default=320)
      parser.add_argument("--height", "-h", type=int, default=240)
      parser.add_argument("--hperiod", "-l", type=float, default=100)
      parser.add_argument("--vperiod", "-m", type=float, default=None)
      parser.add_argument("--octaves", "-k", type=int, default=1)
      parser.add_argument("--persistence", "-p", type=float, default=0.5)
      parser.add_argument("--seed", "-s", type=int, default=None)
      parser.add_argument("--verbose", "-v", action="count", default=0)
      parser.add_argument("--help", "-?", action="help")
      return parser.parse_args()


def main(args: argparse.Namespace) -> None:
      if args.vperiod is None:
            args.vperiod = args.hperiod

      if args.seed is None:
            args.seed = random.randint(1000, 9999)
      rng = np.random.default_rng(args.seed)

      if args.out_tmpl is None:
            args.out_tmpl = f"out/{args.seed}/%.png"
      dirname = os.path.dirname(args.out_tmpl)
      if '%' in dirname:
            raise ValueError("Cannot use index substitution in output dirname")
      os.makedirs(dirname, exist_ok=True)

      gen_times = []
      for i in tqdm.trange(args.num_images, disable=(args.verbose < 1 or args.verbose > 2)):
            start = time.monotonic()
            img = fractal2d(args.width, args.height,
                            args.hperiod, args.vperiod,
                            args.octaves, args.persistence,
                            rng)
            gen_times.append(1000 * (time.monotonic() - start))
            img = (127 * (img + 1)).astype(np.uint8)
            out_path = args.out_tmpl.replace('%', f"{i:02d}")
            imageio.imwrite(out_path, img)
            if args.verbose > 2:
                  print(f"{out_path} written (gen. {gen_times[-1]:.1f}ms).")
      if args.verbose > 1:
            print(f"Compile + first gen. time: {gen_times[0]:.1f}ms")
            avg = np.mean(gen_times[1:])
            print(f"Average gen. time: {avg:.1f}ms")


if __name__ == "__main__":
      main(parse_args())