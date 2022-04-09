
def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
def sort(lista):
  l1=[]
  for i in lista:
    x=i.split('.')
    # y=x[0].split('#')
    l1.append(int(x[0]))
  l1.sort()
  l2=[]
  for i in l1:
    #l2.append('crop#'+str(i)+'.png')
     l2.append(str(i)+'.jpg')
  return l2