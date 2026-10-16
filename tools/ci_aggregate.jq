# Aggregate logic for the required "*-ok" CI jobs.
#
# Input: the job's `needs` context as JSON (`jq -e -f tools/ci_aggregate.jq`).
# Without a "changes" job (ci-ok, security-ok) every job must have succeeded. With a
# "changes" job (sitl-ok, hardware-ok) "changes" must have succeeded, and another job
# may be skipped only when the path filter's own output says there was nothing to run
# (needs.changes.outputs.run == "false"). A missing output never allows a skip.
if has("changes") then
  .changes.result == "success" and
  ((.changes.outputs.run // "") as $run |
    all(to_entries[] | select(.key != "changes");
        .value.result == "success" or
        (.value.result == "skipped" and $run == "false")))
else
  length > 0 and all(.[]; .result == "success")
end
