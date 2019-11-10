junction1_path=$(pwd)/data/real_data/vocabulary-set-b/Junction1
run_bundle=$(pwd)/academic-bundle-2013-07-16-17_24_06/pre-built-release/run-main
junction1_bg=$junction1_path/Junction1.png
juntion1_pwz=$junction1_path/Junction1-b-s-m.tdoc_100.Pwz
results=$(pwd)/data/inference/results
junction1_map=$junction1_path/Junction1.map


mkdir ./data/gifs
if [ -z "$1" ] 
then

   mkdir ./data/gifs/0
   $run_bundle motif-images ./data/gifs/0/junction $junction1_bg $juntion1_pwz $results $junction1_map --simple --rg -m -scale 1
   google-chrome ./data/gifs/0/junction.xhtml

else

   mkdir ./data/gifs/$1
   $run_bundle motif-images ./data/gifs/$1/junction $junction1_bg $juntion1_pwz $results $junction1_map --simple --rg -m -scale 1
   google-chrome ./data/gifs/$1/junction.xhtml

fi
