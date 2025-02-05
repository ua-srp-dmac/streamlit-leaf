while getopts "c:d:m:r:" opt; do
  case ${opt} in
    c)
      run_on_cyverse=$OPTARG
      echo "Running on cyverse: $run_on_cyverse"
      ;;
    d)
      data_path=$OPTARG
      echo "Data path: $data_path"
      ;;
    m)
      model_path=$OPTARG
      echo "Model path: $model_path"
      ;;
    r)
      results_path=$OPTARG
      echo "Results path: $results_path"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

args="$data_path $model_path $results_path"
if [ -n "$run_on_cyverse" ]; then
  args="$args $run_on_cyverse"
fi

exec streamlit run /app/🏠_Home.py --server.enableCORS false --server.enableXsrfProtection false -- $args