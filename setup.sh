echo -e "Please enter your kth username: "
read username

kinit -f $username@KTH.SE
aklog -c kth.se
klist -Tf

ln -s /afs/kth.se/misc/csc/dept/tmh/corpora/timit/timit