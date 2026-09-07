# Publishing releases

1. Set the package version in `pyproject.toml` and finalize the changelog.
2. Commit the release changes, push the branch, and wait for its CI checks.
3. Create the matching `v<version>` tag on that commit and push that specific
   tag. In SourceTree, select only the intended release tag.
4. The tag workflow validates the version, tests and builds the package,
   publishes the distributions to PyPI using trusted publishing, then creates
   the GitHub release with generated notes and the same distribution files.

Ordinary branch pushes do not publish. Tags must exactly match the package
version. Development and local versions are rejected. Alpha, beta, and release
candidate versions use canonical suffixes such as `0.2.0rc1`; their GitHub
releases are marked as prereleases and are not marked Latest. Stable releases
use GitHub's automatic Latest selection. Existing tags are never moved.

If PyPI succeeds but the `github-release` job fails, choose **Re-run failed
jobs** on that workflow run. This retries GitHub release creation without
uploading to PyPI again. The job reuses an existing release and uploads only
missing assets, preserving existing notes and assets. Do not rerun all jobs
or delete and recreate a published tag to retry GitHub release creation.
GitHub limits workflow reruns to 30 days after the original run; recover older
runs manually using the already-published distributions.

The existing PyPI trusted publisher and environment approvals still apply.
The publishing job only needs OIDC and read permissions; the separate GitHub
release job has repository contents write permission. No personal token is
required. A manually created GitHub release does not trigger PyPI publication.
