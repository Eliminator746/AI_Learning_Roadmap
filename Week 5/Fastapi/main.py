from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Item CRUD API")


# output/response model
class Item(BaseModel):
    id:int
    name:str
    description:str
    price:float = Field(gt=0)
    in_stock: bool = True

# data coming from the client    
class ItemCreate(BaseModel):
    name:str
    description:Optional[str] = None
    price:float = Field(gt=0)
    in_stock: bool = True
    


# ---------- "DB" ----------
db: dict[int, Item] = {}
next_id = 1


@app.post("/create-item", response_model=Item)    
async def create_item(item:ItemCreate):
    global next_id
    
    new_item = Item(
        id=next_id,
        name=item.name,
        description=item.description,
        price=item.price,
        in_stock=item.in_stock
    )
    
    db[next_id] = new_item
    
    next_id+=1
    return db

@app.get("/items", response_model=Item[])
async def get_item(item_name:str):
    
    