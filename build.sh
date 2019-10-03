wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1Kcnru7L-iMs3uDRZt3Y0bbkYgMLloM7F' -O trashes.zip
mkdir trashes
unzip -qq -n trashes.zip
cd trashes && rm -rf __MACOSX
cd ../
rm trashes.zip
