task=('C2Q' 'I2Q' 'P2Q' 'R2Q' 'S2Q' 'C2R' 'I2R' 'P2R' 'Q2R' 'S2R' 'C2P' 'I2P' 'Q2P' 'R2P' 'S2P' 'C2S' 'I2S' 'P2S' 'Q2S' 'R2S' 'C2I' 'P2I' 'Q2I' 'R2I' 'S2I' 'I2C' 'P2C' 'Q2C' 'R2C' 'S2C')
source=('clipart' 'infograph' 'painting' 'real' 'sketch' \
        'clipart' 'infograph' 'painting' 'quickdraw' 'sketch' \
        'clipart' 'infograph' 'quickdraw' 'real' 'sketch' \
        'clipart' 'infograph' 'painting' 'quickdraw' 'real' \
        'clipart' 'painting' 'quickdraw' 'real' 'sketch' \
	'infograph' 'painting' 'quickdraw' 'real' 'sketch' )
target=('quickdraw' 'quickdraw' 'quickdraw' 'quickdraw' 'quickdraw' \
        'real' 'real' 'real' 'real' 'real' \
        'painting' 'painting' 'painting' 'painting' 'painting' \
        'sketch' 'sketch' 'sketch' 'sketch' 'sketch' \
        'infograph' 'infograph' 'infograph' 'infograph' 'infograph' \
	'clipart' 'clipart' 'clipart' 'clipart' 'clipart')

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
    s_dset_path='../DomainNet/'${source[index]}'.txt'
    t_dset_path='../DomainNet/'${target[index]}'.txt'
    output_dir='DN_ViT/'${aug_flac}'_'${cdan_flac}'_'${kd_flac}'_'${gsde_flac}'/'${seed[l_id]}'/'${task[index]}
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
