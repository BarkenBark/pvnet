python tools/train_ica_full.py --classname=decilitermatt --resumeTraining |& tee -a decilitermattLog.txt
echo "sleeping for 2 minutes"
sleep 2m
python tools/train_ica_full.py --classname=konserv --resumeTraining |& tee -a konservLog.txt
echo "sleeping for 2 minutes"
sleep 2m
python tools/train_ica_full.py --classname=kaffemugg --resumeTraining |& tee -a kaffemuggLog.txt
echo "sleeping for 2 minutes"
sleep 2m
python tools/train_ica_full.py --classname=tvalgron --resumeTraining |& tee -a tvalgronLog.txt
echo "sleeping for 2 minutes"
sleep 2m
python tools/train_ica_full.py --classname=makaroner --resumeTraining |& tee -a makaronerLog.txt
echo "sleeping for 2 minutes"
sleep 2m
python tools/train_ica_full.py --classname=linser --resumeTraining |& tee -a linserLog.txt
