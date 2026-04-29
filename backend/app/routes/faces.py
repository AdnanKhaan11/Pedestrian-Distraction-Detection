"""
Face API routes.

GET /api/faces — list faces with pagination
DELETE /api/faces/{face_id} — remove face
"""

from typing import Any

from fastapi import APIRouter, Query

from app.database import Database

router = APIRouter(prefix="/api/faces", tags=["Faces"])


@router.get("")
async def list_faces(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(10, ge=1, le=100, description="Results per page"),
) -> dict[str, Any]:
    """
    Get paginated list of all faces.

    Query params:
    - page: Page number (default: 1)
    - limit: Results per page (default: 10, max: 100)

    Returns:
        {
            "total": total count,
            "page": current page,
            "limit": results per page,
            "faces": [
                {
                    "face_id": "...",
                    "first_seen": "ISO8601",
                    "last_seen": "ISO8601",
                    "detection_count": 5,
                    "violation_count": 3,
                    "image_base64": "..."
                },
                ...
            ]
        }
    """
    try:
        db = Database.get_database()

        # Get total count
        total = await db.faces.count_documents({})

        # Calculate skip
        skip = (page - 1) * limit

        # Get faces
        faces_cursor = (
            db.faces.find(
                {}, {"embedding": 0, "detection_ids": 0}
            )  # Exclude large fields
            .sort("last_seen", -1)
            .skip(skip)
            .limit(limit)
        )

        faces = await faces_cursor.to_list(length=limit)

        # Convert _id to face_id in response
        face_list = []
        for face in faces:
            face["face_id"] = face.pop("_id")
            face_list.append(face)

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "faces": face_list,
        }

    except Exception as e:
        return {
            "error": f"Failed to fetch faces: {str(e)}",
            "total": 0,
            "page": page,
            "limit": limit,
            "faces": [],
        }


@router.delete("/{face_id}")
async def delete_face(face_id: str) -> dict[str, Any]:
    """
    Delete a face record by ID.

    Args:
        face_id: UUID string of face to delete

    Returns:
        { "success": true/false, "message": "..." }
    """
    try:
        db = Database.get_database()

        result = await db.faces.delete_one({"_id": face_id})

        if result.deleted_count > 0:
            return {"success": True, "message": f"Face {face_id} deleted"}
        else:
            return {"success": False, "message": f"Face {face_id} not found"}

    except Exception as e:
        return {"success": False, "message": f"Delete failed: {str(e)}"}
