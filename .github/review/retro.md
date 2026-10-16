# Review instructions: retrospective review

An addition to `common.md` for a review of code that is **already merged**, for example
the pull requests that went in before the review gate existed.

- Review the tree at a given commit as if it were one pull request against an empty
  tree, limited to the files the listed pull requests touched. Use the same role
  instructions, severities and output format as for a pull request.
- `Commit:` is the commit whose tree you reviewed (normally `main` at the time).
- For each finding, name the pull request that introduced it when you can tell.
- Merged code cannot be "blocked". `request_changes` here means: the findings need a
  fix. Each `blocker` or `major` finding gets fixed in a normal pull request (or by the
  ticket that owns the code), and the retrospective review is repeated on the new
  commit until all three roles approve.
- The first line is `retro/<role>: approve` or `retro/<role>: request_changes`
  (`<role>` is `architecture`, `security` or `flight-safety`), never `review/...`, so a
  retrospective verdict can never count for `gate/review`. It is posted as a comment
  on the pull request that carries the review (or an issue), not on the old merged
  pull requests.
