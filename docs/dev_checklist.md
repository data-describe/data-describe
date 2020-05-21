# Unit Testing
## Data
Data are currently stored on Google Drive in the interim ([here](https://drive.google.com/drive/folders/1t1B3zzRMSAPsKf6nHtOSfzlvT9_SViU4?usp=sharing)). You must download these files into the `/data` folder in the project directory.

## Running Unit Tests

```console
docker-compose build test
docker-compose run test
```


# Adding a feature
If a new package dependency is introduced, update the following:

-  Update package requirements in all docker/\*/\*.yml
-  Update package requirements in requirements.txt
-  Update package requirements in setup.py

# Submitting a Pull Request
-  [Unit tests run](#running-unit-tests) with > 80% coverage
-  Associated notebook (if applicable) runs with no errors. Unexpected warnings may pass PR (reviewer will decide if it is acceptable or not) but should be added as a new issue to Bitbucket.
    - For example, if there is a single DeprecationWarning raised, that may pass PR and be logged as an issue. On the other hand, if there are hundreds of warnings raised, then that would be considered a blocking issue and should be fixed before submission.

# Releasing new version
Create a new branch from master to address any potential issues and prepare for the version update. In the meantime, **do not allow any pull requests to merge to master**.

-  [Unit tests run](#running-unit-tests) with > 80% coverage    
-  Requirements are fully captured (see [Adding a feature](#adding-a-feature))
-  Version number incremented in `setup.py`
-  All notebooks re-run and checked for errors
-  Update the Sphinx documentation:

```console
python docs/make.py
```
    
-  Build the Datalab image and confirm it runs locally

```console
docker-compose build datalab
docker-compose run -p "127.0.0.1:8081:8080" datalab
```
    
-  Compile release notes as `docs/v<version>_release_notes.md`. You can view what was changed by reviewing commits to master
-  Open a new PR to master and merge when approved
-  Tag the latest commit on master with the version number

```git
git checkout master
git pull
git tag <version>
git push --follow-tags
```
    
-  Tag and push the new Datalab image

```docker
docker tag mw-data_describe_datalab gcr.io/mwpmltr/mw-data_describe:<version>
docker push gcr.io/mwpmltr/mw-data_describe:<version>
```
    

    