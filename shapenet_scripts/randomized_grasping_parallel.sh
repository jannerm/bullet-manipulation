for i in {1..20}
do
	python3 widow_randomized_scripted_grasping.py --data_save_directory WidowGrasp --num_trajectories 100 --num_timesteps 70 &
	sleep 1
done
wait

python3 combine_trajectories.py --data_directory WidowGrasp/trajectories --output_directory WidowGrasp/consolidated_trajectories
