clear all;
close all;
clc;

DB=readtable('titanic.csv');
%Elección de datos
Sur=DB(:,2);
DB=[DB(:,3) DB(:,5) DB(:,6) DB(:,7) DB(:,8) DB(:,10) DB(:,12)];
s=size(DB,1);
for i = 1:s
    if strcmp(DB{i,2},'female')
        A(i,1)=0;
    else
        A(i,1)=1;
    end
end
DB{:,8}=A;
DB.Properties.VariableNames([8]) = {'Gen'};
%Atributos elegidos
DB=[DB(:,1) DB(:,8) DB(:,3) DB(:,4) DB(:,5) DB(:,6)];
%Imputación de datos
A=DB{:,3};
A(isnan(A))=[];
Exc=round(mean(A));
%%
for i=1:size(DB{:,3},1)
    if isnan(DB{i,3})
        DB{i,3}=Exc;
    end
end
%Normalización
DB=table2array(DB);
DB=minMaxNorm(DB);
%Cálculo y orden de eigenvalores 
C=cov(DB);
[V,D]=eig(transpose(C));

E=eig(transpose(C));
[E,I]=sort(E,'descend');


Sur=table2array(Sur);
sinPCA=[Sur, DB];
conPCA=[Sur DB(:,1) DB(:,2) DB(:,3)];

%80% de entrenamiento y 20% de evaluación
sinPCATrain=sinPCA(1:713,:);
sinPCATest=sinPCA(714:end,:);

conPCATrain=conPCA(1:713,:);
conPCATest=conPCA(714:end,:);

%Ejecución de KNN
classSin=KNN(sinPCATrain,sinPCATest,7);
classCon=KNN(conPCATrain,conPCATest,7);

%Evaluación de clasificación
confMatSin=confMatHD(sinPCATest(:,1),transpose(classSin));
[AccS,PreS,SenS,F1ScS] = MatEval(confMatSin);

confMatCon=confMatHD(conPCATest(:,1),transpose(classCon));
[AccC,PreC,SenC,F1ScC] = MatEval(confMatCon);

