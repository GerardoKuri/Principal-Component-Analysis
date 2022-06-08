
%Funci�n la cual recibe un conjunto de entrenamiento etiquetado, un
%conjunto de evaluci�n y la cantidad k de vecinos a considerar.
%Regresa una rreglo respectivo a las predicciones que hace por medio del
%algoritmo knn.

function class=KNN(train,test,k)
S=size(test,1);
cant=size(train,1);
class=[];
for i=1:S
    for j=1:cant
        D(i,j) = sqrt(sum(([test(i,2:end)] - [train(j,2:end)]).^2));
    end
    [x,I]=sort(D(i,:),'ascend');
    t=train(I,:);
    y=mean(t(1:k,1));
    if y>0.5
        class(i)=1;
    elseif y<0.5
        class(i)=0;
    else
        class(i)=t(1);
    end
end
end
