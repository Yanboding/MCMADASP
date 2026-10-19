# MCMADASP project instructions

Before preparing, syncing, submitting, monitoring, or downloading experiments, read and follow [NIBI_EXPERIMENT_WORKFLOW.md](NIBI_EXPERIMENT_WORKFLOW.md). It records the user's default workflow for this project.

Key requirements: use Git for code only; use Nibi with human-approved Duo Push to **Pixel 8 root**; put every command in `table.dat`; clear the four old farm entries only after prior work is inactive and needed outputs are preserved; submit with `submit.run <number_of_jobs> -auto`; keep total account Gurobi token use **strictly below 5000**; download results instead of committing them to Git.

A request for instructions or inspection alone does not authorize submission or cleanup. Follow the full workflow for details, including token accounting and handling `TMP` symlinks.

Local changes must remain uncommitted by default. Before any commit, review whether each change is necessary and whether existing functions, arguments and workflows can be reused. Do not automatically commit, push or sync after editing; follow the latest user instruction in Section 2 of NIBI_EXPERIMENT_WORKFLOW.md.
