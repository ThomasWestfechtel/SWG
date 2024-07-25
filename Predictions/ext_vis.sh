source=('source' 'target')

dset='visda'
index=$1
dom_name=${source[index]}
temp_id='3'
inter='1'

s_fold=('train' 'validation')

dset_path='../../VisDA/'${s_fold[index]}'/image_list.txt'

python ext_zs.py \
       --dset ${dset} \
       --dset_path ${dset_path} \
       --dom_name ${dom_name} \
       --template_id ${temp_id} \
       --interpolate ${inter}
