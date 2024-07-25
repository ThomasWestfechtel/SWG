source=('Art' 'Clipart' 'Product' 'RealWorld')

dset='office-home'
index=$1
dom_name=${source[index]}
temp_id='3'
inter='1'

dset_path='../../OfficeHome/'${source[index]}'/label_c.txt'
python3 ext_zs.py \
       --dset ${dset} \
       --dset_path ${dset_path} \
       --dom_name ${dom_name} \
       --template_id ${temp_id} \
       --network ResNet \
       --interpolate ${inter}
