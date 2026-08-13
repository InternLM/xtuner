#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ] || [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 <pr-number> <output-directory>" >&2
  exit 2
fi

: "${GH_TOKEN:?GH_TOKEN must be set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

pr_number=$1
output_dir=$2
base_git_ref=refs/remotes/origin/claude-review-base
merge_base_git_ref=refs/remotes/origin/claude-review-merge-base
head_git_ref=refs/remotes/origin/claude-review-pr

mkdir -p "$output_dir"
pr_path="$output_dir/pr.json"
diff_path="$output_dir/pr.diff"
diff_index_path="$output_dir/diff-index.txt"
files_path="$output_dir/files.jsonl"
discussion_path="$output_dir/discussion.jsonl"
discussion_index_path="$output_dir/discussion-index.tsv"
manifest_path="$output_dir/manifest.json"
rm -f "$manifest_path"

# Pin all code inputs to one PR revision. Discussions are sampled while the bundle is prepared.
gh pr view "$pr_number" --repo "$GITHUB_REPOSITORY" \
  --json number,title,body,author,url,isDraft,labels,createdAt,updatedAt,baseRefName,baseRefOid,headRefName,headRefOid,additions,deletions,changedFiles \
  > "$pr_path"
base_branch=$(jq -r '.baseRefName' "$pr_path")
base_sha=$(jq -r '.baseRefOid' "$pr_path")
head_branch=$(jq -r '.headRefName' "$pr_path")
head_sha=$(jq -r '.headRefOid' "$pr_path")
merge_base_sha=$(
  gh api "repos/$GITHUB_REPOSITORY/compare/${base_sha}...${head_sha}" --jq '.merge_base_commit.sha'
)

# Fetch the captured SHAs rather than moving branch tips so every local ref belongs to this snapshot.
git fetch --no-tags --depth=1 origin \
  "+${base_sha}:${base_git_ref}" \
  "+${merge_base_sha}:${merge_base_git_ref}" \
  "+refs/pull/${pr_number}/head:${head_git_ref}"
if [ "$(git rev-parse "$base_git_ref")" != "$base_sha" ] || \
   [ "$(git rev-parse "$merge_base_git_ref")" != "$merge_base_sha" ] || \
   [ "$(git rev-parse "$head_git_ref")" != "$head_sha" ]; then
  echo "::error::The pull request changed while its Git refs were being prepared"
  exit 1
fi

# Keep GitHub API output and redirection outside the Claude permission boundary.
gh pr diff "$pr_number" --repo "$GITHUB_REPOSITORY" --color=never > "$diff_path"
grep -n '^diff --git ' "$diff_path" > "$diff_index_path"
gh api --paginate "repos/$GITHUB_REPOSITORY/pulls/$pr_number/files?per_page=100" |
  jq -c '.[] | {
    path: .filename,
    previous_path: (.previous_filename // null),
    status: .status,
    additions: .additions,
    deletions: .deletions,
    changes: .changes,
    blob_sha: .sha,
    has_patch: has("patch")
  }' > "$files_path"

# Top-level comments, reviews, and inline comments are separate GitHub resources.
gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$pr_number/comments?per_page=100" |
  jq -c '.[] | {
    kind: "issue_comment",
    id: .id,
    author: .user.login,
    body: .body,
    created_at: .created_at,
    updated_at: .updated_at,
    url: .html_url
  }' > "$discussion_path"
gh api --paginate "repos/$GITHUB_REPOSITORY/pulls/$pr_number/reviews?per_page=100" |
  jq -c '.[] | {
    kind: "review",
    id: .id,
    author: .user.login,
    state: .state,
    body: .body,
    submitted_at: .submitted_at,
    commit_id: .commit_id,
    url: .html_url
  }' >> "$discussion_path"
gh api --paginate "repos/$GITHUB_REPOSITORY/pulls/$pr_number/comments?per_page=100" |
  jq -c '.[] | {
    kind: "review_comment",
    id: .id,
    review_id: .pull_request_review_id,
    reply_to_id: (.in_reply_to_id // null),
    author: .user.login,
    body: .body,
    path: .path,
    start_line: (.start_line // null),
    line: (.line // null),
    original_start_line: (.original_start_line // null),
    original_line: (.original_line // null),
    side: (.side // null),
    commit_id: .commit_id,
    original_commit_id: .original_commit_id,
    created_at: .created_at,
    updated_at: .updated_at,
    url: .html_url
  }' >> "$discussion_path"

# Keep the common deduplication fields compact so review agents only open full discussion records when needed.
jq -nr '
  def normalize_body:
    (. // "")
    | tostring
    | gsub("[\\t\\r\\n]+"; " ")
    | gsub(" {2,}"; " ")
    | sub("^ "; "")
    | sub(" $"; "");

  (["kind", "author", "commit_id", "path", "line", "body"] | @tsv),
  (inputs | [
    (.kind // ""),
    (.author // ""),
    (.commit_id // .original_commit_id // ""),
    (.path // ""),
    (.line // .original_line // .start_line // .original_start_line // ""),
    (.body | normalize_body)
  ] | @tsv)
' "$discussion_path" > "$discussion_index_path"

# Reject a mixed code snapshot if either side moved while the bundle was being built.
if ! gh pr view "$pr_number" --repo "$GITHUB_REPOSITORY" --json baseRefOid,headRefOid |
  jq -e --arg base "$base_sha" --arg head "$head_sha" \
    '.baseRefOid == $base and .headRefOid == $head' > /dev/null; then
  echo "::error::The pull request changed while review inputs were being prepared"
  exit 1
fi

diff_sha256=$(sha256sum "$diff_path" | awk '{print $1}')
file_count=$(wc -l < "$files_path")
discussion_count=$(wc -l < "$discussion_path")
jq -n \
  --arg repository "$GITHUB_REPOSITORY" \
  --argjson pr_number "$pr_number" \
  --arg base_branch "$base_branch" \
  --arg base_sha "$base_sha" \
  --arg base_git_ref "$base_git_ref" \
  --arg merge_base_sha "$merge_base_sha" \
  --arg merge_base_git_ref "$merge_base_git_ref" \
  --arg head_branch "$head_branch" \
  --arg head_sha "$head_sha" \
  --arg head_git_ref "$head_git_ref" \
  --arg pr_path "$pr_path" \
  --arg diff_path "$diff_path" \
  --arg diff_index_path "$diff_index_path" \
  --arg diff_sha256 "$diff_sha256" \
  --arg files_path "$files_path" \
  --argjson file_count "$file_count" \
  --arg discussion_path "$discussion_path" \
  --arg discussion_index_path "$discussion_index_path" \
  --argjson discussion_count "$discussion_count" \
  '{
    schema_version: 1,
    repository: $repository,
    pr_number: $pr_number,
    base: {branch: $base_branch, sha: $base_sha, git_ref: $base_git_ref},
    merge_base: {sha: $merge_base_sha, git_ref: $merge_base_git_ref},
    head: {branch: $head_branch, sha: $head_sha, git_ref: $head_git_ref},
    metadata: {path: $pr_path},
    diff: {path: $diff_path, index_path: $diff_index_path, sha256: $diff_sha256},
    files: {path: $files_path, count: $file_count},
    discussion: {path: $discussion_path, index_path: $discussion_index_path, count: $discussion_count}
  }' > "$manifest_path"
