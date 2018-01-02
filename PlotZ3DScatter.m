%%
% Do a 3D scatter plot for each value in Z, using different color for each
% class.
%
% Input
% - Z (N x 4): Output from hidden units.
% - y (N x 1): Predicted labels (0 to 9)
%
function PlotZ3DScatter(Z,y)

plt = Z(:,2:end);
[N,D] = size(y);
labels = zeros(N,1);
for x=1:N
    [val,idx] = max(y(x,:));
    labels(x) = idx;
end
rgb = labels.*1;
scatter3(plt(:,1),plt(:,2),plt(:,3),10,rgb,'filled');
text(plt(:,1), plt(:,2),plt(:,3), cellstr(num2str(labels)))
%%%%
end

