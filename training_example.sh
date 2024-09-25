# Sample shell code to train an MTP instance using the training dataset

# Copy untrained mtp files from MLIP-2 installation, replace species count to 3 (CrCoNi)
POT=`sed PATH/TO/YOUR/MLIP-2/INSTALLATION/untrained_mtps/20.mtp -e "s/species_count = 1/species_count = 3/g"` 
printf "${POT}" > Potentials/init_20.mtp

# Document training time
SECONDS=0

# Perform training with mpi parallelzation
mpirun -n 64 mlp train Potentials/init_20.mtp Training_datasets/Training_Cao_20220823.cfg --curr-pot-name=Potentials/cur_20.mtp --trained-pot-name=Potentials/trained_20.mtp --max-iter=10000 --bfgs-conv-tol=1e-5 --update-mindist

# Report training time
duration=$SECONDS
echo "Total job time: $duration seconds"