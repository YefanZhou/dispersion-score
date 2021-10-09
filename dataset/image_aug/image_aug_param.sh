#for mag in 0 3 5 7 9
for nop in 0 1 2 3 4
do 
    ##Mag
    # python image_aug_param.py  --c_method KMedoids --e_method Inertia \
    #        --n_cluster 5 --aug-type augment_mag --n_op 2 --mag_idx $mag \
    #         --prob 1 --nepoch 1 --sample_num 30 --perceptual

    ##NOP
    python image_aug_param.py  --c_method KMedoids --e_method Inertia \
           --n_cluster 5 --aug-type augment_mag --n_op $nop --mag_idx 4 \
            --prob 0.5 --nepoch 1 --sample_num 30 --perceptual 
    # python image_aug_param.py  --c_method KMedoids --e_method Inertia \
    #        --n_cluster 5 --aug-type augment_mag --n_op $nop --mag_idx 4 \
    #         --prob 1 --nepoch 1 --sample_num 30 --perceptual

    ##Auto Aug
    # python image_aug_param.py  --c_method KMedoids --e_method Inertia \
    #        --n_cluster 5 --aug-type auto_aug --sample_num 30 
    # python image_aug_param.py  --c_method KMedoids --e_method Inertia \
    #        --n_cluster 5 --aug-type auto_aug --perceptual --sample_num 30 
           
done




