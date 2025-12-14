function resultat = PuntsDintre(P, N)

if nargin < 2, N = 250; end

% 1. Generem punts aleatoris

xp = -50 + 100*rand(N,1);   % entre -50 i 50
yp = -35 + 70*rand(N,1);    % entre -35 i 35

% 2. Mirem si estan dintre o fora del poligon

[in, on] = inpolygon(xp, yp, P(:,1), P(:,2));

% 3. Contem els punts del perímetre com a punts interiors

insideFlag = double(in | on);


resultat = [xp, yp, insideFlag];

figure; hold on; axis equal;

% Dibuxa el poligon
fill(P(:,1), P(:,2), [0.2 0.6 1], 'FaceAlpha',0.3, 'EdgeColor','k');

% Dibuixa els punts: Dintre = Vert, Fora = Vermell
plot(xp(insideFlag==1), yp(insideFlag==1), 'gx', 'MarkerSize',8, 'LineWidth',1.5);
scatter(xp(insideFlag==0), yp(insideFlag==0), 20, 'r', 'filled');

% Llegenda
legend('Polígon','Punts a dintre','Punts a fora');
xlabel('x'); ylabel('y');
title('Polígon i punts aleatoris per entrenar');
hold off;

% Formategem i guardem les dades en un archiu csv per després llegir-los
% amb el programa de càrrega de dades en python

T = array2table(resultat, ...
    'VariableNames', {'x','y','Sortida'});

Nom = 'dades.csv';

writetable(T, Nom);

end