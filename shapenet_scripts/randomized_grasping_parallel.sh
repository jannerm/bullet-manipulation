for i in {1..20}
do
	python3 randomized_scripted_grasping.py --data-save-directory SawyerGrasp --num-trajectories 10 --num-timesteps 50 &
	sleep 2
done
wait

python3 combine_trajectories.py --data-directory SawyerGrasp/trajectories --output-directory SawyerGrasp/consolidated_trajectories --pool-type TrajectoryPool
