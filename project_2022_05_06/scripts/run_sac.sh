
set -e
set -x

# ENV=HalfCheetah-v2
ENV=Humanoid-v2
# ENV_TYPE=mb
ENV_TYPE=sac
SEED=300
NUM_ENV=1


if [ "$(uname)" == "Darwin" ]; then
  python3 -m sac.main -s \
    algorithm=sac \
    seed=${SEED} \
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type=${ENV_TYPE}
elif [ "$(uname)" == "Linux" ]; then
  # for ENV in HalfCheetah-v2 Walker2d-v2
  for ENV in Humanoid-v2
  do
  for SEED in 100 200 300
    do
      python3 -m sac.main -s \
      algorithm=sac \
      seed=${SEED} \
      env.id=${ENV} \
      env.num_env=${NUM_ENV} \
      env.env_type=${ENV_TYPE} & sleep 2
    done
    wait
  done
fi
