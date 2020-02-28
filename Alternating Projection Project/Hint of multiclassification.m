%A hint of constructing the training matrix input
function[A_tilde, b_tilde] = construct_train(X_tr, y_tr, digit)
[row, col] = size(X_tr);
len = digit* col;
total = 0;
A_tilde = zeros(row*(digit -1), len);
b_tilde = zeros(row*(digit -1),1);

for i = 1:row
	true = y_tr(i) +1;
	for j = 1: digit
		if j == true
			continue;
		else
			temp = zeros(1, len);
			temp((true-1) *col +1 : true *col) = X_tr(i,:);
			temp((j-1)*col + 1:j*col) = -X_tr(i, :)
			A_tilde(total = total + 1;
		end
	end
end
end