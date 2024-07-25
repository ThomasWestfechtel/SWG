task=('A2C' 'A2P' 'A2R' 'C2A' 'C2P' 'C2R' 'P2A' 'P2C' 'P2R' 'R2A' 'R2C' 'R2P')
source=('Art' 'Art' 'Art' 'Clipart' 'Clipart' 'Clipart' 'Product' 'Product' 'Product' 'RealWorld' 'RealWorld' 'RealWorld')
target=('Clipart' 'Product' 'RealWorld' 'Art' 'Product' 'RealWorld' 'Art' 'Clipart' 'RealWorld' 'Art' 'Clipart' 'Product')

meth='CDANE'
dset='office-home'
seed=('42' '43' '44')
index=$1
lr='5e-6'
bs='64'
tau_p='0.90'
epochs='100'
pretext='0'
perc=('0.5')

aug_flac='0'
cdan_flac='1'
kd_flac='1'
gsde_flac='1'

for((l_id=0; l_id < 1; l_id++))
do
for((h_id=0; h_id < 1; h_id++))
do
    echo ">> Seed 42: traning task ${index} : ${task[index]}"
    s_dset_path='../OfficeHome/'${source[index]}'/label_c.txt'
    t_dset_path='../OfficeHome/'${target[index]}'/label_c.txt'
    output_dir='OH_ViT/'${aug_flac}'_'${cdan_flac}'_'${kd_flac}'_'${gsde_flac}'/'${seed[l_id]}'/'${task[index]}
    python main_ViT.py \
       --meth ${meth} \
       --dset ${dset} \
       --s_dset_path ${s_dset_path} \
       --t_dset_path ${t_dset_path} \
       --output_dir ${output_dir} \
       --lr ${lr} \
       --s ${source[index]} \
       --t ${target[index]} \
       --tau_p ${tau_p} \
       --epochs ${epochs[0]} \
       --bs ${bs} \
       --pretext ${pretext} \
       --run_id ${h_id} \
       --perc ${perc[h_id]} \
       --aug_flac ${aug_flac} \
       --cdan_flac ${cdan_flac} \
       --kd_flac ${kd_flac} \
       --gsde_flac ${gsde_flac} \
       --seed ${seed[l_id]} 
done
done
