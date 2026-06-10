#!/bin/bash
# Pilot progress check. Run:  bash /home/jupyter/til-ai-26/asr/finetune/progress.sh
FT=/home/jupyter/til-ai-26/asr/finetune

echo "===== TTS generation ====="
status=$(docker ps -a --filter name=asr-tts-gen --format '{{.Status}}' 2>/dev/null)
echo "container : ${status:-not found}"
clips=$(ls "$FT/synth/clips/" 2>/dev/null | wc -l)
echo "clips     : $clips / 741"
if [ "$clips" -gt 0 ]; then
  pct=$(( clips * 100 / 741 ))
  bar=$(printf '#%.0s' $(seq 1 $(( pct / 4 + 1 ))))
  echo "progress  : [${bar}] ${pct}%"
fi
echo "last log  : $(docker logs asr-tts-gen 2>&1 | grep -E 'sentences|DONE' | tail -1)"

echo
echo "===== augmented clips (step 5, after Gate A) ====="
echo "clips     : $(ls "$FT/synth/clips_aug/" 2>/dev/null | wc -l)"

echo
echo "===== GPU pilot run ====="
ts=$(docker ps -a --filter name=asr-pilot-pipeline --format '{{.Status}}')
if [ -n "$ts" ]; then
  echo "container : $ts"
  logs=$(docker logs asr-pilot-pipeline 2>&1)
  step=$(echo "$logs" | grep -oE '===== [0-9]/3 [^=]+=====' | tail -1)
  echo "stage     : ${step:-(starting)}"
  prog=$(echo "$logs" | tr '\r' '\n' | grep -E 'Epoch [0-9]+: +[0-9]+%.*it/s' | tail -1 | sed 's/^ *//')
  [ -n "$prog" ] && echo "progress  : $prog"
  verdict=$(echo "$logs" | grep -E 'VERDICT:|BEST CKPT:|PIPELINE DONE' | tail -2)
  [ -n "$verdict" ] && echo "result    :" && echo "$verdict"
else
  echo "(not started)"
fi

echo
echo "===== outputs produced ====="
for f in pilot_targets.json pilot_sentences.json synth_manifest.json \
         pronunciation_report.json synth_aug_manifest.json \
         pilot_train_manifest.json; do
  [ -f "$FT/output/$f" ] && echo "  [x] $f" || echo "  [ ] $f"
done
