# DLFinalProject
**On Edge MRI Abnormality Detection using Teacher-Knowledge Distillation**

we propose an on-edge compressed model that can be deployed on mobile devices. We train this student model by transferring dark knowledge from a deeper and complex pre-trained teacher model. The objective is to show that a knowledge-distilled student model (KD), weighing in at a mere 2.5 MB, can accurately detect the presence of a tumor and predict its type based on input brain MRI images, much like the teacher model, which is 130 MB in size. KD student model can be deployed on mobile devices and personalized using continuous training from user data providing personalized solutions. Since MRI analysis requires consultation with doctor in the real world, our suggested approach adds an extra layer of service while cutting down on needless work and expenses. 

Architecture for training Smaller Student Model using Pre-trained Teacher Model.


![Model Architecture](https://github.com/HARSHALK2598/DLFinalProject/assets/59302243/ae6e8b85-6ff9-4f72-bfed-2e186dcd0638)

Reference: Z. Tao, Q. Xia, and Q. Li, “Neuron manifold distillation for edge deep learning,” in Proc. IEEE/ACM 29th Int. Symp. Qual. Service, 2021, pp. 1–10. 


Teacher Model Architecture:

![image](https://github.com/HARSHALK2598/DLFinalProject/assets/59302243/09132a7e-1e67-4e71-87d3-c42c2d3ccd45)


Student Model Architecture:

![image](https://github.com/HARSHALK2598/DLFinalProject/assets/59302243/f93051dc-2d54-43f8-80da-5a4c2b48c34d)


Results folder contains all plots. 


**For runtime Train/Test accuracy and Train/Test loss,** \
please refer KD_student_output_logs.txt for KD_Student Model . \
Baseline Student model: /Student Model/Baseline_Student_logs.txt . \
Teacher Model: /Teacher Model/VGG_teacherModel_output.txt . 


Trained Teacher Model is not uploaded due to size of 60MB.  \
Trained Baseline Student model and KD student model can found in repository.  



