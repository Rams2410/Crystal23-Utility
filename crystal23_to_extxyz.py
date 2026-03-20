"""
Crystal23 raw output → Extended XYZ converter for ML training.

Reads: FORCES.DAT, SCFOUT.LOG, INPUT, OUTPUT, opt[a/c]* files
Writes: Extended XYZ file specified via command-line argument

Usage:
    python crystal23_to_extxyz.py <output_file.xyz>
"""

import sys
import os
import re
import warnings
import logging
import numpy as np

warnings.filterwarnings("ignore")

####### Physical constants #######################################################
HA_BOHR_TO_EV_ANG = 51.422086   #  Ha/Bohr  to  eV/Å
HA_TO_EV          = 27.2114079527  #  Hartree  to  eV

####### Logging setup #######################################################
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


####### Utility helpers #######################################################

def grep_file(pattern: str, filename: str) -> list[list[str]]:
    """Return split tokens of every line in *filename* that matches *pattern*."""
    try:
        with open(filename) as fh:
            return [line.strip().split() for line in fh if re.search(pattern, line)]
    except FileNotFoundError:
        log.error("File '%s' not found.", filename)
        return []


def lines_after_marker(file_path: str, marker: str, n_lines: int) -> list[str]:
    """
    Return up to *n_lines* lines that follow each occurrence of *marker*
    in *file_path*.  All occurrences are collected (matches original behaviour).
    """
    collected: list[str] = []
    try:
        with open(file_path) as fh:
            count = 0
            inside = False
            for line in fh:
                if marker in line:
                    inside = True
                    count = 0
                elif inside:
                    count += 1
                    collected.append(line)
                    if count == n_lines:
                        inside = False
    except FileNotFoundError:
        log.error("File '%s' not found.", file_path)
    return collected


####### Energy parsers #######################################################

def get_energy(energy_type: str, job_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse total and free energies from SCFOUT.LOG (and OUTPUT for geometry jobs).

    Returns
    -------
    total_energies : np.ndarray  (Hartree)
    free_energies  : np.ndarray  (Hartree)
    """
    total_energies: list[float] = []
    free_energies:  list[str]   = []

    # For geometry optimisations grab the *first* SCF energy before the first
    # force calculation from the OUTPUT file.
    if job_type in ("OptGeom", "reOptGeom"):
        try:
            with open("OUTPUT") as fh:
                last_free  = None
                last_total = None
                found      = False
                for line in fh:
                    if r"FREE ENERGY      (F)" in line:
                        last_free = line.strip().split()
                    if "TOTAL ENERGY" in line:
                        last_total = line.strip().split()
                    if ("FORCE CALCULATION" in line
                            and last_free is not None
                            and last_total is not None
                            and not found):
                        free_energies.append(last_free[3])
                        total_energies.append(float(last_total[3]))
                        found = True
                        break
        except FileNotFoundError:
            log.warning("OUTPUT file not found; skipping initial geometry energy.")

    # Main energy source: SCFOUT.LOG
    scf_log = "SCFOUT.LOG"
    total_pattern = re.compile(r"TOTAL ENERGY.*?([-+]?\d+\.\d+E[+-]\d+)")
    free_pattern  = r"FREE ENERGY      "

    try:
        with open(scf_log) as fh:
            for line in fh:
                m = total_pattern.search(line)
                if m:
                    total_energies.append(float(m.group(1)))
    except FileNotFoundError:
        log.error("SCFOUT.LOG not found.")

    for row in grep_file(free_pattern, scf_log):
        free_energies.append(row[3])

    free_arr  = np.asarray(free_energies,  dtype=np.float64)
    total_arr = np.asarray(total_energies, dtype=np.float64)

    log.info("Free-energy values parsed from SCFOUT: %d", len(free_arr))
    return total_arr, free_arr


def get_opt_energies(calc_type: str, job_type: str) -> tuple[list[str], np.ndarray, int]:
    """
    Collect energies from opt[a/c]* files sorted numerically.

    Returns
    -------
    opt_files      : list of filenames (sorted)
    opt_energies   : np.ndarray
    dimensionality : int  (2 or 3)
    """
    log.info("Scanning for opt files: calc_type=%s, job_type=%s", calc_type, job_type)

    opt_files = sorted(
        [f for f in os.listdir() if calc_type in f],
        key=lambda f: int(re.sub(calc_type, "", f)),
    )

    if not opt_files:
        raise FileNotFoundError(f"No opt files matching '{calc_type}*' found.")

    opt_energies:  list[float] = []
    dimensionality = 3

    for fname in opt_files:
        with open(fname) as fh:
            tokens = fh.readline().strip().split()
            dimensionality = int(tokens[0])
            opt_energies.append(float(tokens[4]))

    log.info("Opt-file energies parsed: %d", len(opt_energies))
    return opt_files, np.asarray(opt_energies, dtype=np.float64), dimensionality


####### Lattice / PBC helpers #######################################################

def pbc_string(dim: int) -> str:
    return "T T F" if dim == 2 else "T T T"


def parse_lattice(opt_file: str) -> np.ndarray:
    """Read the 3×3 lattice matrix (Å) from the lines after 'DE' in an opt file."""
    raw = lines_after_marker(opt_file, "DE", 3)
    stripped = [l.strip() for l in raw]
    return np.asarray([s.split() for s in stripped], dtype=np.float64)


####### Extended-XYZ builder #######################################################

def build_extxyz_lines(
    forces_file: str,
    total_energies: np.ndarray,
    free_energies:  np.ndarray,
    opt_files:      list[str],
    opt_energies:   np.ndarray,
    dim:            int,
) -> tuple[list[str], int]:
    """
    Parse FORCES.DAT and cross-reference energies to assemble Extended-XYZ lines.

    Returns
    -------
    lines     : list[str]   ready to write
    n_written : int         number of structures included
    """
    output_lines: list[str] = []
    n_written = 0
    force_energy_count = 0

    with open(forces_file) as fh:
        all_lines = fh.readlines()

    n_atoms = int(all_lines[0].strip())
    log.info("Atoms per cell: %d", n_atoms)

    inside_block = False
    skip         = True
    count        = 0
    free_idx     = -1

    lattice_str = ""
    pbc         = pbc_string(dim)

    for line in all_lines:
        if "DE" in line:
            tokens = line.strip().split()
            force_energy_count += 1

            # Locate the energy value that follows the 'E' token
            try:
                e_pos = tokens.index("E")
                ff    = float(tokens[e_pos + 1])
            except (ValueError, IndexError):
                log.warning("Could not parse energy from line: %s", line.strip())
                inside_block = False
                skip         = True
                continue

            # Cross-check with SCFOUT total energies
            scf_hits = np.where(total_energies == ff)[0]
            if scf_hits.size == 0:
                skip = True
                inside_block = True
                count = 0
                continue

            free_idx = int(scf_hits[0])

            # Cross-check with opt-file energies
            opt_hits = np.where(opt_energies == ff)[0]
            if opt_hits.size == 0:
                skip = True
                inside_block = True
                count = 0
                continue

            opt_idx   = int(opt_hits[0])
            lattice   = parse_lattice(opt_files[opt_idx])
            free_en   = float(free_energies[free_idx]) * HA_TO_EV

            a, b, c   = lattice[0], lattice[1], lattice[2]
            lattice_str = (
                f"{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} "
                f"{b[0]:.4f} {b[1]:.4f} {b[2]:.4f} "
                f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
            )

            output_lines.append(f"{n_atoms}\n")
            output_lines.append(
                f'Lattice="{lattice_str}" '
                f"Properties=species:S:1:pos:R:3:forces:R:3:resid:I:1 "
                f"energy={free_en} "
                f'pbc="{pbc}"\n'
            )

            skip         = False
            inside_block = True
            count        = 0
            n_written   += 1

        elif inside_block and not skip:
            count += 1
            parts = line.rstrip("\n").split()
            if len(parts) != 7:
                log.warning("Unexpected atom line (skipping): %s", line.strip())
                continue

            element, x, y, z, fx, fy, fz = parts
            element = element.capitalize()

            x, y, z = float(x), float(y), float(z)
            fx = float(fx) * HA_BOHR_TO_EV_ANG
            fy = float(fy) * HA_BOHR_TO_EV_ANG
            fz = float(fz) * HA_BOHR_TO_EV_ANG

            output_lines.append(
                f"{element:2} "
                f"{x:>20.16f} {y:>20.16f} {z:>20.16f} "
                f"{fx:>20.16f} {fy:>20.16f} {fz:>20.16f} "
                f"{count}\n"
            )

            if count == n_atoms:
                inside_block = False
                skip         = True

    log.info("Force-file energy blocks scanned: %d", force_energy_count)
    return output_lines, n_written


####### Main #######################################################

def main() -> None:
    if len(sys.argv) < 2:
        log.error("Usage: python crystal23_to_extxyz.py <output_file>")
        sys.exit(1)

    output_path = sys.argv[1]

    # ── Determine job configuration from INPUT #######################################################
    is_restart = bool(grep_file("RESTART",  "INPUT"))
    has_smear  = bool(grep_file("SMEAR",    "INPUT"))
    is_atomonly= bool(grep_file("ATOMONLY", "INPUT"))

    job_type    = "reOptGeom" if is_restart else "OptGeom"
    energy_type = "Free"      if has_smear  else "Total"
    calc_type   = "opta"      if is_atomonly else "optc"

    log.info("Job type: %s | Energy type: %s | Opt files: %s*",
             job_type, energy_type, calc_type)

    ######## Parse energies #######################################################
    total_energies, free_energies = get_energy(energy_type, job_type)
    opt_files, opt_energies, dim  = get_opt_energies(calc_type, job_type)

    log.info("Dimensionality: %dD", dim)

    ######## Build output #######################################################
    lines, n_structures = build_extxyz_lines(
        "FORCES.DAT",
        total_energies,
        free_energies,
        opt_files,
        opt_energies,
        dim,
    )

    ######## Write (overwrite if exists) #######################################################
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "w") as fh:
        fh.writelines(lines)

    log.info("Structures extracted: %d", n_structures)
    log.info("Output written to: %s", output_path)
    print("\nJob done! Bye.")


if __name__ == "__main__":
    main()
