while getopts "p:" opt; do
  case ${opt} in
    p)
      streamlit_path=$OPTARG
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

echo "$streamlit_path"

exec streamlit run /app/🏠_Home.py $streamlit_path --server.enableCORS false --server.enableXsrfProtection false