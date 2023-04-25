mkdir ./temp
mkdir ./temp/code
cp results/model.pt ./temp
cp model.py ./temp/code
cp inference.py ./temp/code
cd ./temp
tar -czvf model.tar.gz *
mv model.tar.gz ../results/zip
cd ../
rm -rf ./temp





