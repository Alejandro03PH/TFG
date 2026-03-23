function V = Generar_Poligon(numVert, numSamples, turnVar, minEdge, maxEdge)

%   Genera un poligon aleatori
%
% Inputs:
%   numVert    - Nombre de vertex
%   numSamples - Nombre de punts del perímetre que retorna
%   turnVar    - Variació mitja de canvi d'angle entre cada vector de dos
%   punts
%   minEdge    - Mínima distància entre dos punts del perímetre
%   maxEdge    - Màxima distància entre dos punts del perímetre
%
% Output:
%   V - numSamples punts del perimetre en una matriu numSamples x 2

    if nargin < 5, maxEdge = 1.3; end
    if nargin < 4, minEdge = 0.9; end
    if nargin < 3, turnVar = 0.05; end
    if nargin < 2, numSamples = 200; end
    if nargin < 1, numVert = 200; end


    angles = zeros(numVert,1);
    for i = 2:numVert
        angles(i) = angles(i-1) + sqrt(turnVar)*randn;
    end

    step = 1;
    dx = step * cos(angles);
    dy = step * sin(angles);

    % polygon vertices
    V_orig = [cumsum(dx), cumsum(dy)];
    V_orig = V_orig - mean(V_orig,1);

    ang = atan2(V_orig(:,2), V_orig(:,1));
    [~, order] = sort(ang);
    V_sorted = V_orig(order,:);

    diffs = diff([V_sorted; V_sorted(1,:)],1,1);
    edgeLen = sqrt(sum(diffs.^2,2));
    keepIdx = true(size(V_sorted,1),1);

    for i = 1:length(V_sorted)
        nextEdge = edgeLen(i);
        if nextEdge > maxEdge
            keepIdx(i) = false;  % remove spike
        end
        if nextEdge < minEdge
            keepIdx(i) = true;   % preserve small curves
        end
    end
    V_clean = V_sorted(keepIdx,:);

    diffs = diff([V_clean; V_clean(1,:)],1,1);
    edgeLen = sqrt(sum(diffs.^2,2));
    cumLen = [0; cumsum(edgeLen)];
    perim = cumLen(end);

    spacing = perim / numSamples;
    s = 0:spacing:(perim-spacing);
    V = zeros(numSamples,2);

    for k = 1:numSamples
        idx = find(cumLen <= s(k),1,'last');
        if idx == length(edgeLen)+1, idx = 1; end
        frac = (s(k)-cumLen(idx))/edgeLen(idx);
        V(k,:) = V_clean(idx,:) + frac*diffs(idx,:);
    end


    % Plot

    figure; hold on; axis equal;
    fill(V(:,1), V(:,2), [0.85 0.9 1], 'EdgeColor','k','LineWidth',1.5);
end
