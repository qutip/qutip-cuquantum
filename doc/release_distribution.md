# Release Procedure for qutip-cuquantum

#### Major and Minor Releases (e.g., v0.1.0, v0.2.0)

- Create a new branch named qutip-cuquantum-A.B.X, where A.B is the major/minor version number.

- Update the VERSION file for Release. Remove the .dev suffix from the version number. `alpha`, `beta`, `post` tag are acceptable.

- Increment Version on main. Edit the VERSION file and increment the version number, resetting the dev number. For example, if you are releasing `v0.1.0`, the version on main should become `v0.2.0.dev1`.


#### Micro/Patch Releases (e.g., v0.1.1)

- Make a branch off the release branch.
- Cherry-Pick changes to include from main.
- Increment the VERSION file
- Merge into the release branch with a pull request.

#### Deployment

- Go to the "Releases" page on the GitHub repository and draft a new release.
- Create a new tag with the scheme vA.B.C (e.g., v0.1.0) targeting your release branch (qutip-cuquantum-0.1.X).
- Write the changelog into the release description.
- Publishing the release will trigger the GitHub Action to build and publish the package to PyPI.
