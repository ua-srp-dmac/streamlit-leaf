# streamlit-leaf

Streamlit app for leaf segmentation.

To run locally:

I recommend to create a project folder `streamlit-leaf-seg` that will contain 3 directories:

1. this github repository (`streamlit-leaf`)
2. directory containing ML model files (`models`)
3. directory containing data (`data`)


So first, create the directory for the project, create the `data` and `models` folders, and clone the git repository into the project folder.
```
$ mkdir streamlit-leaf-seg 
$ cd streamlit-leaf-seg
$ mkdir data
$ mkdir models
$ git clone git@github.com:michellito/streamlit-leaf.git
```

Then, populate your folders with these example data/models stored on [CyVerse here](https://de.cyverse.org/data/ds/iplant/home/shared/srp_dmac/dmac/rhizobox?selectedOrder=asc&selectedOrderBy=name&selectedPage=0&selectedRowsPerPage=100). You'll need to create a CyVerse account with an .edu email and request access to view the folder.  Please refer to [Transferring Data with Cyberduck](https://learning.cyverse.org/ds/cyberduck/) to download data from CyVerse.

Once you have your folders populated, you can build and run the docker container from the `streamlit-leaf` directory. You'll need to replace `<your-dockerhub-username>` with your own username and replace the `/path/to/data` and `path/to/models` with the full path to the data/models folders that you just created.

```
$ cd streamlit-leaf
$ docker build -t <your-dockerhub-username>/streamlit-leaf:latest .

$ docker run --rm -e PYTHONUNBUFFERED=1 -v /path/to/data:/cyverse/data -v /path/to/models:/cyverse/models -w /cyverse/data -p 8501:8501 <your-dockerhub-username>/streamlit-leaf:latest -p /cyverse/
```

The streamlit app can then be accessed at `http://localhost:8501`.