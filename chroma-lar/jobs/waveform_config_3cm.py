import os

"""
10M: mean = 6.356 s, std = 0.811 s --> 1596672 vox = 2,819 hours 
15M: mean = 9.168 s, std = 0.779 s --> 1596672 vox = 4,066 hours
20M: mean = 11.957 s, std = 1.190 s --> 1596672 vox = 5,303 hours
25M: mean = 14.880 s, std = 0.727 s --> 1596672 vox = 6,599.5776 hours
"""

# Detector and voxel configuration
config = {
    # Detector dimensions (mm)
    "detector_x_range": (-2310, 0),  # Only simulating half (mirrored)
    "detector_y_range": (-2160, 2160),
    "detector_z_range": (-2160, 2160),
    # Voxel parameters
    "voxel_size": 30,  # mm
    # Simulation parameters
    "nphotons": 15_000_000,
    "detector_config": "detector_config_reflect_reflect3wires",
    # Job parameters
    "time_per_voxel": 10,  # seconds per voxel
    "max_job_time": 8 * 60 * 60,  # 12 hours in seconds
    "slurm_max_job_time_buffer": 2 * 60 * 60,  # 2 hours in seconds
    # Site-specific configuration
    "site": "perlmutter",
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
        output_dir=f"/global/cfs/cdirs/m5238/users/{os.environ['USER']}/prod_chroma_lar/waveform_map",
        slurm=dict(
            account='m5238_g',
            output=f"/global/cfs/cdirs/m5238/users/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            error=f"/global/cfs/cdirs/m5238/users/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            qos='shared',
            constraint='gpu',
        )
    )
)
