for i in {1..20}
do
	python3 randomized_scripted_grasping.py --data_save_directory SawyerGrasp --num_trajectories 100 --num_timesteps 50 &
	sleep 1
done
wait

python3 combine_trajectories.py --data_directory SawyerGrasp/trajectories --output_directory SawyerGrasp/consolidated_trajectories
