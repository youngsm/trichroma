import os

"""
10M: mean = 6.356 s, std = 0.811 s
15M: mean = 9.168 s, std = 0.779 s
20M: mean = 11.957 s, std = 1.190 s
25M: mean = 14.880 s, std = 0.727 s
"""
# Detector and voxel configuration
config = {
    # Detector dimensions (mm)
    "detector_x_range": (-2310, 0),  # Only simulating half (mirrored)
    "detector_y_range": (-2160, 2160),
    "detector_z_range": (-2160, 2160),
    # Voxel parameters
    "voxel_size": 300,  # mm
    # Simulation parameters
    "nphotons": 15_000_000,
    "detector_config": "detector_config_reflect_reflect3wires",
    # Job parameters
    "time_per_voxel": 9.168,  # seconds per voxel
    "max_job_time": 3 * 60 * 60,  # 3 hours in seconds
    "slurm_max_job_time_buffer": 2 * 60,  # 2 minutes in seconds
    "site": "slac",
}


site = dict(
    slac=dict(
        work_dir="$LSCRATCH",
        container_cmd="singularity exec --nv -B /lscratch,/sdf /sdf/home/y/youngsam/sw/dune/sim/chroma-lar/installation/chroma3.lar-plib/chroma.simg",
        output_dir=f"/sdf/data/neutrino/{os.environ['USER']}/prod_chroma_lar/waveform_map",
        slurm=dict(
            partition='ampere',
            account='neutrino:cider-nu',
            output=f"/sdf/data/neutrino/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            error=f"/sdf/data/neutrino/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
        ),
    ),
    perlmutter=dict(
        work_dir="$PSCRATCH",
        container_cmd="shifter --image=deeplearnphysics/simlar:larchroma-2025-07-18",
        output_dir=f"/global/cfs/cdirs/dune/users/{os.environ['USER']}/prod_chroma_lar/waveform_map",
        slurm=dict(
            account='dune',
            output=f"/global/cfs/cdirs/dune/users/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            error=f"/global/cfs/cdirs/dune/users/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            qos='shared',
            constraint='gpu',
        )
    )
)
