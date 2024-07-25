source=('clipart' 'infograph' 'painting' 'quickdraw' 'real' 'sketch') 

dset='domain-net'
index=$1
dom_name=${source[index]}
temp_id='3'
inter='1'

dset_path='../../DomainNet/'${source[index]}'.txt'
python ext_zs.py \
       --dset ${dset} \
       --dset_path ${dset_path} \
       --dom_name ${dom_name} \
       --template_id ${temp_id} \
       --interpolate ${inter}
