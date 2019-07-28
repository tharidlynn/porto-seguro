tmp = time.time()
assert len(X_train) == len(y_train)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, random_state=SEED, test_size=0.3)

print('{} is training', MODEL_NAME)
eval_set = [(X_tr, y_tr), (X_val, y_val)]
model.fit(X_tr, y_tr, eval_set=eval_set, eval_metric=gini.gini_xgbsklearn, early_stopping_rounds=100, verbose=True)

# y_len = [y for y in y_val if y==1]
# print(len(y_len))
# tn, fp, fn, tp = confusion_matrix(y_val, model.predict(X_val)).ravel()
# print('Confusion Matrix: tn, fp, fn, tp')
# print(tn, fp, fn, tp)
gc.collect()

oof_test = model.predict_proba(X_test)[:,1]
# Export oof_test
file_path = os.path.join(OOF_PATH, MODEL_NAME + '_test.csv')
pd.DataFrame({MODEL_NAME: oof_test}).to_csv(file_path, index=False)
# np.savetxt(file_path, oof_test.reshape(-1, 1), delimiter=',', fmt='%.5f')

print('SUCCESSFULLY SAVE {} AT {}  PLEASE VERIFY THEM'.format(MODEL_NAME, OOF_PATH))
print('Training time: {} minutes'.format(str((time.time() - tmp) / 60)))
