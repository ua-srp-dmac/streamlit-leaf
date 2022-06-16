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

ls /

exec streamlit run /app/app.py $streamlit_path --server.enableCORS false --server.enableXsrfProtection false